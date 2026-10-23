"""Bad answers of the model under a JSON response format must end in the structured-output error, never in a dropped connection, a mislabelled
400 or a stream without a terminal event (A07b): an answer nested deeper than the decoder allows (RecursionError) and a valid-JSON escape that is
a lone surrogate (the canonical text could not be sent as UTF-8).  Real handlers, scripted fake engine, CPU only.

The two POLICY tests pin current behaviour that is an open question (an array holding an object is delivered as that object)."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import server, structured
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}, "s": {"type": "string"}}, "required": ["a"]}
MSGS = [{"role": "user", "content": "x"}]


def depth_that_overflows():
    n = 2000
    while n <= 400000:
        try:
            json.loads('{"a":' * n + "1" + "}" * n)
        except RecursionError:
            return n
        n *= 2
    return None


class Unit(unittest.TestCase):
    def setUp(self):
        self.validators = {"object": structured.prepare_format({"type": "json_object"}, MSGS)[1],
                           "schema": structured.prepare_format({"type": "json_schema", "json_schema": {"name": "n", "schema": SCHEMA}}, MSGS)[1]}

    def test_too_deep_is_a_structured_output_error(self):
        n = depth_that_overflows()
        if n is None:
            self.skipTest("this Python never overflows the JSON decoder")
        for name, v in self.validators.items():
            with self.subTest(validator=name), self.assertRaises(structured.StructuredOutputError):
                structured.validated_json('{"a":' * n + "1" + "} " * n, v, "stop")

    def test_a_lone_surrogate_is_a_structured_output_error(self):
        for escape in ("\\ud800", "\\udfff", "x\\udc00y"):
            for name, v in self.validators.items():
                with self.subTest(escape=escape, validator=name), self.assertRaises(structured.StructuredOutputError) as c:
                    structured.validated_json('{"a": 1, "s": "' + escape + '"}', v, "stop")
                self.assertIn("surrogate", str(c.exception))

    def test_a_pair_of_surrogates_and_other_unicode_are_still_delivered(self):
        v = self.validators["schema"]
        self.assertEqual(structured.validated_json('{"a": 1, "s": "\\ud83d\\ude00 \\u00e9 中"}', v, "stop"), '{"a":1,"s":"\U0001f600 é 中"}')
        self.assertEqual(structured.validated_json('{"a": 1}', v, "stop"), '{"a":1}')

    def test_POLICY_an_array_holding_an_object_is_delivered_as_the_object(self):
        """_extract_json takes the first balanced {...} wherever it is: [{"a":1}] -> {"a":1} (and prose or a fence around it is dropped)"""
        for name, v in self.validators.items():
            self.assertEqual(structured.validated_json('[{"a": 1}, {"a": 2}]', v, "stop"), '{"a":1}')
            self.assertEqual(structured.validated_json('Here: {"a": 1} ok {"a": 2}', v, "stop"), '{"a":1}')
        for text in ("[1, 2]", "42", "null", ""):
            with self.assertRaises(structured.StructuredOutputError):
                structured.validated_json(text, self.validators["object"], "stop")


class Tok(ByteTokenizer):
    def __init__(self):
        self.extra = []

    def token_bytes(self, t):
        return self.extra[t - 512] if t >= 512 else self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])

    def decode(self, ids, errors="replace"):
        return b"".join(self.token_bytes(t) for t in ids).decode("utf-8", errors)


class Script(MockEngine):
    def __init__(self, tok, ids):
        super().__init__(tok, "x", 4_000_000)
        self.ids, self.closed = ids, False
        self.max_context = self.known_ctx = 4_000_000

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            for t in (self.ids + [257])[:max_new]:
                if cancel.is_set():
                    return
                yield t
        finally:
            self.closed = True


def ask(answer: bytes, route, stream):
    tok = Tok()
    eng = Script(tok, list(b"</think>\n\n" + answer))
    server.Service.start_telemetry = lambda self: None
    svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=0)
    try:
        if route == "chat":
            path, body = "/v1/chat/completions", {"model": "x", "messages": MSGS, "stream": stream, "max_tokens": 3_000_000,
                                                  "response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": SCHEMA}}}
        else:
            path, body = "/v1/responses", {"model": "x", "input": "x", "stream": stream, "max_output_tokens": 3_000_000,
                                           "text": {"format": {"type": "json_schema", "name": "n", "schema": SCHEMA}}}
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()
    finally:
        httpd.shutdown()
        httpd.server_close()


def data_events(raw):
    return [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: {")]


class Handlers(unittest.TestCase):
    ANSWERS = {"lone surrogate": b'{"a": 1, "s": "\\ud800"}'}

    def check_failed(self, route, stream, status, raw):
        if not stream:
            self.assertEqual(status, 502, raw[:200])
            err = json.loads(raw)["error"]
            self.assertEqual(err.get("code") or err.get("type"), "structured_output_failed")
        elif route == "chat":
            self.assertEqual(status, 200)
            events = data_events(raw)
            self.assertEqual([e["error"].get("code") for e in events if "error" in e], ["structured_output_failed"])
            self.assertTrue(raw.rstrip().endswith("[DONE]"))
            self.assertFalse([e for e in events if e.get("choices") and e["choices"][0]["delta"].get("content")])
        else:
            events = data_events(raw)
            types = [e["type"] for e in events]
            self.assertEqual(types[-1], "response.failed", types)
            self.assertEqual(events[-1]["response"]["error"]["code"], "structured_output_failed")
            self.assertNotIn("response.output_text.delta", types)
            self.assertNotIn("response.completed", types)

    def test_lone_surrogate_on_both_apis_stream_and_not(self):
        for route in ("chat", "responses"):
            for stream in (False, True):
                with self.subTest(route=route, stream=stream):
                    self.check_failed(route, stream, *ask(self.ANSWERS["lone surrogate"], route, stream))

    def test_too_deep_answer_on_both_apis(self):
        n = depth_that_overflows()
        if n is None or n > 100000:
            self.skipTest("no usable overflow depth on this Python")
        answer = b'{"a":' * n + b"1" + b"} " * n                 # alternating tokens: a run of one token would end the fake script at 256
        for route, stream in (("chat", False), ("responses", True)):
            with self.subTest(route=route, stream=stream):
                self.check_failed(route, stream, *ask(answer, route, stream))

    def test_a_valid_answer_is_still_delivered_once(self):
        status, raw = ask(b'{"a": 1, "s": "\\ud83d\\ude00"}', "chat", False)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], '{"a":1,"s":"\U0001f600"}')
        status, raw = ask(b'{"a": 1}', "responses", True)
        events = data_events(raw)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual([e["delta"] for e in events if e["type"] == "response.output_text.delta"], ['{"a":1}'])


if __name__ == "__main__":
    unittest.main()
