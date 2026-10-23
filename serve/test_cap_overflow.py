"""serve/test_cap_overflow.py - a reply cap that is not a finite number (A07a-F8), through the real HTTP handler and a recording fake engine.

JSON `1e309` (and the non-standard literals Infinity / -Infinity that Python's decoder also reads) is infinity.  Chat and Messages
converted the cap with int() and the OverflowError closed the connection before any answer; every such request is now a 400
invalid_request_error that names the field, with no engine call, no event stream and a healthy next request.  The normal coercions,
sentinels, precedence and the context check stay as they were (the positive controls)."""
import contextlib
import http.client
import io
import unittest
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

CTX = 4096


class RecEngine(MockEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = []                                  # the max_new of every generate() call

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.seen.append(max_new)
        return super().generate(ids, max_new, sampling, cancel, embeddings)


def chat(**fields):
    body = '{"messages":[{"role":"user","content":"hi"}]' + "".join(f',"{k}":{v}' for k, v in fields.items()) + "}"
    return "/v1/chat/completions", body


def messages(**fields):
    body = '{"messages":[{"role":"user","content":"hi"}]' + "".join(f',"{k}":{v}' for k, v in fields.items()) + "}"
    return "/v1/messages", body


def responses(**fields):
    body = '{"input":"hi"' + "".join(f',"{k}":{v}' for k, v in fields.items()) + "}"
    return "/v1/responses", body


ROUTES = {"chat/max_tokens": (chat, "max_tokens"), "chat/max_completion_tokens": (chat, "max_completion_tokens"),
          "messages/max_tokens": (messages, "max_tokens"), "responses/max_output_tokens": (responses, "max_output_tokens")}


class Http(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        Service.start_telemetry = lambda self: None
        self.eng = RecEngine(tok, "hello world", max_context=CTX)
        self.svc = Service(self.eng, tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.addCleanup(lambda: (self.httpd.shutdown(), self.httpd.server_close()))

    def post(self, path, raw):
        with contextlib.redirect_stdout(io.StringIO()):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            try:
                c.request("POST", path, body=raw.encode(), headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                r = c.getresponse()
                return r.status, r.getheader("Content-Type") or "", r.read().decode()
            except (http.client.RemoteDisconnected, ConnectionError) as e:
                return None, type(e).__name__, ""
            finally:
                c.close()

    def healthy(self):
        n = len(self.eng.seen)
        status, _, _ = self.post(*chat(max_tokens=2))
        self.assertEqual(status, 200)
        self.assertEqual(len(self.eng.seen), n + 1)


class NonFiniteCaps(Http):
    def check_rejected(self, route, path, raw, field, literal):
        n = len(self.eng.seen)
        status, ctype, body = self.post(path, raw)
        self.assertEqual(status, 400, (route, literal, ctype))                   # was: the connection closed without an answer
        self.assertIn("application/json", ctype, (route, literal))
        self.assertIn("invalid_request_error", body if route != "responses/max_output_tokens" else "invalid_request_error" + body, (route, literal, body))
        self.assertEqual(len(self.eng.seen), n, (route, literal))                 # nothing reached the engine
        self.assertIn(field, body, (route, literal, body))

    def test_infinity_in_every_cap_field_is_a_400_that_names_the_field(self):
        for route, (build, field) in ROUTES.items():
            for literal in ("1e309", "-1e309", "Infinity", "-Infinity"):
                for stream in ("false", "true"):
                    path, raw = build(**{field: literal, "stream": stream})
                    with self.subTest(route=route, literal=literal, stream=stream):
                        self.check_rejected(route, path, raw, field, literal)
        self.healthy()

    def test_nan_is_a_400_as_before(self):
        for route, (build, field) in ROUTES.items():
            path, raw = build(**{field: "NaN"})
            with self.subTest(route=route):
                status, _, body = self.post(path, raw)
                self.assertEqual(status, 400, body)
                self.assertEqual(len(self.eng.seen), 0)
        self.healthy()

    def test_the_monitor_finishes_the_record_of_a_rejected_request(self):
        for route, (build, field) in ROUTES.items():
            path, raw = build(**{field: "1e309"})
            before = len(self.svc.request_records(None))
            self.post(path, raw)
            records = self.svc.request_records(None)
            self.assertEqual(len(records), before + 1, route)
            rec = records[0]
            self.assertEqual(rec.get("http_status"), 400, route)
            self.assertTrue(rec.get("finished_at"), route)
            self.assertEqual(rec.get("state"), "error", route)

    def test_chat_precedence_between_the_two_fields_is_kept(self):
        path, raw = chat(max_completion_tokens="1e309", max_tokens=5)
        status, _, body = self.post(path, raw)
        self.assertEqual(status, 400)
        self.assertIn("max_completion_tokens", body)                          # the field that decides is the one named
        self.assertEqual(self.eng.seen, [])
        path, raw = chat(max_completion_tokens=0, max_tokens="1e309")        # 0 is no cap: max_tokens decides
        status, _, body = self.post(path, raw)
        self.assertEqual(status, 400)
        self.assertIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)
        path, raw = chat(max_completion_tokens=5, max_tokens="1e309")        # the cap that decides is fine: the other is not read
        status, _, _ = self.post(path, raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.eng.seen, [5])

    def test_a_shared_default_does_not_hide_a_bad_field_and_fills_an_absent_one(self):
        self.svc.shared = {"max_tokens": 7}
        status, _, _ = self.post(*chat())
        self.assertEqual((status, self.eng.seen), (200, [7]))                # absent: the shared setting
        path, raw = chat(max_tokens="1e309")
        status, _, body = self.post(path, raw)
        self.assertEqual(status, 400)                                         # present: the request's own value wins, and is checked
        self.assertIn("max_tokens", body)
        self.assertEqual(self.eng.seen, [7])


class Unchanged(Http):
    def rest(self):
        """the 'rest of the context' for this prompt, as the unchanged code reaches it from a 0 cap"""
        self.post(*chat(max_tokens=0))
        return self.eng.seen[-1]

    def test_normal_caps_and_sentinels(self):
        rest = self.rest()
        self.assertTrue(0 < rest < CTX)
        want = {"5": 5, "5.9": 5, '"7"': 7, "true": 1, "0": rest, "-1": rest, "null": rest, "false": rest}
        for route, (build, field) in ROUTES.items():
            if route.startswith("responses"):
                continue                                                      # Responses is strict (a positive whole number) and stays so
            for literal, expected in want.items():
                with self.subTest(route=route, literal=literal):
                    n = len(self.eng.seen)
                    status, _, body = self.post(*build(**{field: literal}))
                    self.assertEqual(status, 200, body)
                    self.assertEqual(self.eng.seen[n:], [expected])

    def test_a_finite_cap_beyond_the_context_is_still_the_context_check(self):
        for route, (build, field) in ROUTES.items():
            if route.startswith("responses"):
                continue
            for literal in ("4096", "100000", "1000000000000000000000000000000"):
                with self.subTest(route=route, literal=literal):
                    status, _, body = self.post(*build(**{field: literal}))
                    self.assertEqual(status, 400)
                    self.assertIn("exceed", body)
                    self.assertEqual(self.eng.seen, [])

    def test_responses_stays_strict(self):
        for literal, ok in (("5", True), ("0", False), ("-1", False), ('"7"', False), ("5.9", False), ("1e309", False)):
            with self.subTest(literal=literal):
                status, _, _ = self.post(*responses(max_output_tokens=literal))
                self.assertEqual(status == 200, ok)

    def test_a_non_numeric_string_is_still_a_400(self):
        status, _, body = self.post(*chat(max_tokens='"abc"'))
        self.assertEqual(status, 400)
        self.assertIn("invalid_request_error", body)


if __name__ == "__main__":
    unittest.main()
