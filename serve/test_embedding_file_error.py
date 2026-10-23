"""A04a (2): a local file error while the images' combined embeddings file is made or written is the server's, before any
answer: HTTP 500 `server_error` in the API's own error shape - not a connection that just closes - with the half-written
file and the thread's path/keys cleaned up, no engine started and a clean next request.  A client that really left, and
every other OSError, are treated as before.

    python -m unittest serve.test_embedding_file_error -v
"""
from __future__ import annotations

import errno
import hashlib
import http.client
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import serve.server as server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

PNG = "data:image/png;base64,iVBORw0KGgo="
SECRET = "/dev/shm/strata-secret-path-A04"


class StubVision:
    """The encoder as the service sees it: a directory, and one embeddings file (3 tokens) per distinct image source."""

    def __init__(self, d):
        self.dir = Path(d)
        self.elastic = False
        self.encoded = []

    def encode(self, source, keep=()):
        key = hashlib.sha256(source.encode()).hexdigest()[:32]
        f = self.dir / f"{key}.sve"
        f.write_bytes(b"SVE0" + key.encode())
        self.encoded.append(source)
        return f, 3

    def encode_all(self, sources, cancel=None):
        return [self.encode(s) for s in sources]


def chat(image=PNG):
    return "/v1/chat/completions", {"max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "S1"}, {"type": "image_url", "image_url": {"url": image}}]}]}


def messages(image=PNG):
    return "/v1/messages", {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "S1"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image.split(",", 1)[1]}}]}]}


def responses(image=PNG):
    return "/v1/responses", {"model": "m", "max_output_tokens": 16, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "S1"}, {"type": "input_image", "image_url": image}]}]}


APIS = {"chat": chat, "messages": messages, "responses": responses}


def disk_full(path, parts):
    Path(path).write_bytes(b"half")                       # a file that is half written when the error comes
    raise OSError(errno.ENOSPC, "No space left on device", SECRET)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = mock.patch.object(Service, "start_telemetry", lambda self: None)
        cls.patch.start()
        cls.tmp = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        cls.vision = StubVision(cls.tmp.name)
        cls.engine = MockEngine(tok, "hello", max_context=65536)
        cls.calls = []
        real = cls.engine.generate
        cls.engine.generate = lambda ids, max_new, sampling, cancel, embeddings=None: (
            cls.calls.append({"emb": Path(embeddings).read_bytes() if embeddings else None}) or real(ids, max_new, sampling, cancel, embeddings))
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=cls.vision)
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.shm0 = cls.shm()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()
        cls.patch.stop()

    @staticmethod
    def shm():
        return {p.name for p in Path("/dev/shm").glob(f"strata-vision-{os.getpid()}-*")}

    def left_over(self):
        mine = [n for n in self.shm() - self.shm0]
        return mine + [p.name for p in Path(self.tmp.name).glob("req-*")]

    def post(self, path, body, raw=False):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            return r.status, r.getheader("Content-Type") or "", data if raw else json.loads(data or b"{}")
        finally:
            c.close()

    def setUp(self):
        self.calls.clear()


class FileErrorTests(Base):
    def check_500(self, api, stream, patcher):
        path, body = APIS[api]()
        if stream:
            body["stream"] = True
        with mock.patch("builtins.print"), patcher:
            status, ctype, raw = self.post(path, body, raw=True)
        text = raw.decode("utf-8", "replace")
        out = json.loads(text)
        self.assertEqual(status, 500, text)
        self.assertIn("application/json", ctype)                  # an error answer, not the start of a stream
        err = out["error"]
        self.assertEqual(err["type"], "server_error")
        self.assertNotIn("data:", text.splitlines()[0])
        for leak in (SECRET, "/dev/shm", "No space left", "ENOSPC", self.tmp.name, "strata-vision"):
            self.assertNotIn(leak, text)                          # no path, no system text
        if api == "responses":
            self.assertEqual(set(err) >= {"message", "type", "code"}, True)
        self.assertEqual(self.calls, [])                          # no engine started
        self.assertEqual(self.left_over(), [])                    # the half-written file is gone

    def test_a_write_error_is_a_500_on_every_api_streamed_or_not(self):
        for api in APIS:
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    self.check_500(api, stream, mock.patch.object(server, "write_temporary", disk_full))

    def test_the_combined_file_cannot_be_placed(self):
        def broken(vision_dir, nbytes):
            raise OSError(errno.EACCES, "Permission denied", SECRET)
        for api in APIS:
            with self.subTest(api=api):
                self.check_500(api, False, mock.patch.object(server, "combined_embeddings_path", broken))

    def test_a_cached_embeddings_file_that_is_gone(self):
        """encode() answers a file that has vanished: the size lookup fails before anything is written."""
        gone = self.vision.dir / "gone.sve"
        for api in APIS:
            with self.subTest(api=api):
                self.check_500(api, False, mock.patch.object(self.vision, "encode_all", lambda sources, cancel=None: [(gone, 3)]))

    def test_the_next_request_is_clean(self):
        for api in APIS:
            with self.subTest(api=api):
                with mock.patch("builtins.print"):
                    self.check_500(api, False, mock.patch.object(server, "write_temporary", disk_full))
                    path, body = APIS[api](PNG + "x")
                    status, _, out = self.post(path, body)
                self.assertEqual(status, 200, out)
                self.assertEqual(len(self.calls), 1)
                self.assertTrue(self.calls[0]["emb"].startswith(b"SVE0"))     # its own embeddings, none of the failed request's
                self.assertEqual(self.left_over(), [])
                self.calls.clear()

    def test_the_thread_state_is_cleared(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "S1"}, {"type": "image", "source": PNG}]}]
        with mock.patch.object(server, "write_temporary", disk_full), mock.patch("builtins.print"):
            with self.assertRaises(server.EmbeddingFileError) as cm:
                self.svc.prepare(msgs, None, {}, 4)
        self.assertIsNone(getattr(self.svc.embeddings, "path", None))
        self.assertIsNone(getattr(self.svc.embeddings, "keys", None))
        self.assertIsInstance(cm.exception.__cause__, OSError)           # the cause stays for the log
        self.assertEqual(self.left_over(), [])


class DoubleFaultTests(Base):
    """The write fails AND the half-written file cannot be removed (the same full or read-only place): the 500 must still go
    out, the thread's state must be clear, and the leftover is a limit that is reported, not a deletion that is claimed."""

    real_unlink = Path.unlink

    def refuse_unlink(self):
        real = type(self).real_unlink

        def unlink(path, missing_ok=False):
            if path.name.startswith(("strata-vision-", "req-")):
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
            return real(path, missing_ok=missing_ok)
        return mock.patch.object(Path, "unlink", unlink)

    def remove_leftovers(self):
        for n in self.left_over():
            for d in (Path("/dev/shm"), Path(self.tmp.name)):
                (d / n).unlink(missing_ok=True)

    def tearDown(self):
        self.remove_leftovers()

    def logged(self, fn):
        lines = []
        with mock.patch("builtins.print", lambda *a, **k: lines.append(" ".join(str(x) for x in a))):
            fn()
        return "\n".join(lines)

    def test_the_500_goes_out_when_the_cleanup_fails_too(self):
        for api in APIS:
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    path, body = APIS[api]()
                    if stream:
                        body["stream"] = True
                    result = {}
                    def go():
                        result["r"] = self.post(path, body, raw=True)
                    with mock.patch.object(server, "write_temporary", disk_full), self.refuse_unlink():
                        log = self.logged(go)
                    status, ctype, raw = result["r"]
                    text = raw.decode("utf-8", "replace")
                    self.assertEqual(status, 500, text)
                    self.assertIn("application/json", ctype)
                    self.assertEqual(json.loads(text)["error"]["type"], "server_error")
                    for leak in (SECRET, "/dev/shm", "No space left", "Permission denied", "EACCES", "ENOSPC", self.tmp.name, "strata-vision"):
                        self.assertNotIn(leak, text)
                    self.assertEqual(self.calls, [])
                    # the log tells both causes by type and errno - and no path
                    self.assertIn(f"OSError, errno {errno.ENOSPC}", log)
                    self.assertIn(f"PermissionError, errno {errno.EACCES}", log)
                    self.assertIn("could not be removed", log)
                    for leak in (SECRET, "/dev/shm", self.tmp.name, "strata-vision"):
                        self.assertNotIn(leak, log)
                    self.assertTrue(self.left_over(), "the file that could not be removed is the limit; it is still there")
                    self.remove_leftovers()

    def test_the_thread_state_is_clear_and_the_cause_is_the_write_error(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "S1"}, {"type": "image", "source": PNG}]}]
        with mock.patch.object(server, "write_temporary", disk_full), self.refuse_unlink(), mock.patch("builtins.print"):
            with self.assertRaises(server.EmbeddingFileError) as cm:       # not the PermissionError of the cleanup
                self.svc.prepare(msgs, None, {}, 4)
        self.assertIsNone(getattr(self.svc.embeddings, "path", "unset"))
        self.assertIsNone(getattr(self.svc.embeddings, "keys", "unset"))
        self.assertEqual(cm.exception.__cause__.errno, errno.ENOSPC)

    def test_the_next_request_is_clean_when_the_space_is_back(self):
        path, body = chat()
        with mock.patch.object(server, "write_temporary", disk_full), self.refuse_unlink(), mock.patch("builtins.print"):
            status, _, _ = self.post(path, body)
        self.assertEqual(status, 500)
        self.remove_leftovers()                                               # the place is writable again
        self.calls.clear()
        path, body = chat(PNG + "again")
        with mock.patch("builtins.print"):
            status, _, out = self.post(path, body)
        self.assertEqual(status, 200, out)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0]["emb"].startswith(b"SVE0"))
        self.assertEqual(self.left_over(), [])

    def test_a_successful_cleanup_is_not_reported_as_a_leftover(self):
        path, body = chat()
        with mock.patch.object(server, "write_temporary", disk_full):
            log = self.logged(lambda: self.post(path, body))
        self.assertNotIn("could not be removed", log)
        self.assertEqual(self.left_over(), [])


class UnchangedTests(Base):
    """What this fix must not touch."""

    def request_closes_without_answer(self, patcher):
        path, body = chat()
        with mock.patch("builtins.print"), patcher:
            with self.assertRaises((http.client.RemoteDisconnected, ConnectionError)):
                self.post(path, body)

    def test_a_broken_pipe_is_still_a_client_that_left(self):
        self.request_closes_without_answer(mock.patch.object(self.svc, "prepare", side_effect=BrokenPipeError()))

    def test_any_other_oserror_is_treated_as_before(self):
        self.request_closes_without_answer(mock.patch.object(self.svc, "encode_prompt", side_effect=OSError(errno.EIO, "I/O error")))

    def test_an_oserror_outside_the_embeddings_file_is_not_reclassified(self):
        """An OSError from the image encode itself (not the combined file) keeps its old route."""
        self.request_closes_without_answer(mock.patch.object(self.vision, "encode_all", side_effect=OSError(errno.EIO, "boom")))

    def test_a_good_request_with_an_image_answers_200(self):
        for api in APIS:
            with self.subTest(api=api):
                path, body = APIS[api](PNG + "ok")
                with mock.patch("builtins.print"):
                    status, _, out = self.post(path, body)
                self.assertEqual(status, 200, out)
                self.assertEqual(self.left_over(), [])
        self.assertEqual(len(self.calls), 3)

    def test_the_engine_starting_is_still_503_and_writes_nothing(self):
        saved = self.engine.max_context
        self.engine.max_context = 0
        try:
            path, body = chat(PNG + "starting")
            with mock.patch("builtins.print"):
                status, _, out = self.post(path, body)
        finally:
            self.engine.max_context = saved
        self.assertEqual(status, 503, out)
        self.assertEqual(self.left_over(), [])


if __name__ == "__main__":
    unittest.main()
