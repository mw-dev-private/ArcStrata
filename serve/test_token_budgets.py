"""Token budgets from the request to the engine (A07a): the output cap field, the context room, the thinking budget with its answer reserve and
wrap-up, and the budget refusals - through the real HTTP handlers and the real Service.run with a recording fake engine.  CPU only.

Tests marked POLICY pin current behaviour that is an open question (lenient parsing of the cap field, falsy values meaning "no cap",
count_tokens counting one marker per image); they exist so a change is noticed, not because the behaviour is endorsed."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
WRAP = 18                                       # REASONING_WRAP_UP in tokens, as the pack tokenizer counts it (measured)


class Tok(ByteTokenizer):
    """byte tokens, with token_bytes; the wrap-up text is 18 tokens like the pack's"""
    def __init__(self):
        self.extra = []

    def token_bytes(self, t):
        return self.extra[t - 512] if t >= 512 else self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])

    def decode(self, ids, errors="replace"):
        return b"".join(self.token_bytes(t) for t in ids).decode("utf-8", errors)

    def encode(self, text, parse_special=False, plain=()):
        if text == server.REASONING_WRAP_UP:
            b = text.encode()
            cuts = [round(i * len(b) / WRAP) for i in range(WRAP + 1)]
            ids = []
            for a, c in zip(cuts, cuts[1:]):
                if b[a:c] not in self.extra:
                    self.extra.append(b[a:c])
                ids.append(512 + self.extra.index(b[a:c]))
            return ids
        return super().encode(text, parse_special, plain)


def cyc(n, upper=False):
    return list(((b"ABCDEFGHIJKLMNOPQRSTUVWXYZ" if upper else b"abcdefghijklmnopqrstuvwxyz") * (n // 26 + 1))[:n])


class Rec(MockEngine):
    """plays a script per generate() call and records what each call was given"""
    def __init__(self, tok, scripts, ctx):
        super().__init__(tok, "x", ctx)
        self.scripts, self.calls = scripts, []
        self.max_context = self.known_ctx = ctx

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        rec = {"prompt_len": len(ids), "max_new": max_new, "handed": 0}
        self.calls.append(rec)
        script = self.scripts[min(len(self.calls) - 1, len(self.scripts) - 1)] + [257]
        for t in script[:max_new]:
            if cancel.is_set():
                return
            rec["handed"] += 1
            yield t


def service(scripts, ctx=100000, fit=False, shared=None):
    tok = Tok()
    eng = Rec(tok, scripts, ctx)
    server.Service.start_telemetry = lambda self: None
    svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), fit_max_tokens=fit)
    svc.shared = dict(shared or {})
    return svc, eng


class Http:
    def __init__(self, scripts=None, **kw):
        self.svc, self.eng = service(scripts or [cyc(30)], **kw)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()


ROUTES = {"openai": ("/v1/chat/completions", "max_tokens", lambda: {"messages": [{"role": "user", "content": "hi"}]}),
          "anthropic": ("/v1/messages", "max_tokens", lambda: {"messages": [{"role": "user", "content": "hi"}]}),
          "responses": ("/v1/responses", "max_output_tokens", lambda: {"input": "hi"})}


def ask(h, route, cap="__absent__", extra=None):
    path, field, base = ROUTES[route]
    body = {"model": "x", **base(), **(extra or {})}
    if cap != "__absent__":
        body[field] = cap
    n = len(h.eng.calls)
    status, raw = h.post(path, body)
    return status, raw, h.eng.calls[n:]


def prompt_len(route):
    h = Http(ctx=100000)
    try:
        ask(h, route, 4)
        return h.eng.calls[0]["prompt_len"]
    finally:
        h.close()


class ContextRoom(unittest.TestCase):
    """room = ctx - 8 - prompt; cap absent/0/-1 -> the rest; cap > room -> 400 (fit off) / clamped (fit on); every 400 before the engine"""

    def test_room_contract_on_every_route(self):
        for route in ROUTES:
            P = prompt_len(route)
            for fit in (False, True):
                for room in (-1, 0, 1, 2, 5):
                    h = Http(ctx=P + 8 + room, fit=fit)                      # one server per (route, fit, room): the caps below reuse it
                    try:
                        for label, cap in (("absent", "__absent__"), ("1", 1), ("room-1", room - 1), ("room", room), ("room+1", room + 1)):
                            if cap != "__absent__" and cap < 1:
                                continue
                            with self.subTest(route=route, fit=fit, room=room, cap=label):
                                status, raw, calls = ask(h, route, cap)
                                if cap == "__absent__" or cap <= room:
                                    want = ("ok", room if cap == "__absent__" else cap) if room >= 1 else ("400",)
                                else:
                                    want = ("ok", max(1, room)) if (fit and room >= 1) else ("400",)
                                if want[0] == "400":
                                    self.assertEqual((status, len(calls)), (400, 0))
                                else:
                                    self.assertEqual((status, calls[0]["max_new"]), (200, want[1]))
                    finally:
                        h.close()

    def test_a_refusal_names_the_room_and_leaves_the_prompt_cache_clean(self):
        P = prompt_len("openai")
        h = Http(ctx=P + 8 + 3)
        try:
            status, raw, calls = ask(h, "openai", 50)
            self.assertEqual((status, len(calls)), (400, 0))
            self.assertIn("at most 3 here", raw)
            self.assertEqual(h.svc.prompts.entries, [])
            self.assertEqual(ask(h, "openai", 3)[0], 200)
            self.assertEqual(len(h.svc.prompts.entries), 1)
        finally:
            h.close()

    def test_context_sources(self):
        h = Http(ctx=500)
        try:
            h.eng.max_context, h.eng.known_ctx = 0, 500                  # a failed restart: the last known context
            h.eng.alive = lambda: True
            self.assertEqual(ask(h, "openai", 4)[0], 200)
            h.eng.known_ctx, h.eng.starting = 0, True
            self.assertEqual(ask(h, "openai", 4)[0], 503)
            self.assertEqual(h.svc.prompts.entries[0][2][:1] != [], True)   # an engine that is starting keeps the prompt
        finally:
            h.close()


class CapField(unittest.TestCase):
    def test_sentinels_and_priority(self):
        for route in ("openai", "anthropic"):
            for cap in (None, 0, -1, -5):
                with self.subTest(route=route, cap=cap):
                    h = Http(ctx=8192)
                    try:
                        status, raw, calls = ask(h, route, cap)
                        P = calls[0]["prompt_len"]
                        self.assertEqual((status, calls[0]["max_new"]), (200, 8192 - 8 - P))     # the rest of the context
                    finally:
                        h.close()
        h = Http(ctx=8192)
        try:
            for mt, mc, want in ((5, 0, 5), (0, 7, 7), (5, 7, 7), (None, 7, 7), (5, None, 5)):
                status, raw, calls = ask(h, "openai", mt, {"max_completion_tokens": mc})
                self.assertEqual((status, calls[0]["max_new"]), (200, want), (mt, mc))
        finally:
            h.close()

    def test_responses_wants_a_positive_whole_number(self):
        h = Http(ctx=8192)
        try:
            for bad in (0, -1, True, 5.0, 5.5, "7", [], {}):
                status, raw, calls = ask(h, "responses", bad)
                self.assertEqual((status, len(calls)), (400, 0), bad)
                self.assertIn("max_output_tokens", raw)
        finally:
            h.close()

    def test_shared_default_fills_only_where_the_request_has_none(self):
        h = Http(ctx=8192, shared={"max_tokens": 11})
        try:
            for cap, want in (("__absent__", 11), (None, 11), (3, 3)):
                self.assertEqual(ask(h, "openai", cap)[2][0]["max_new"], want, cap)
        finally:
            h.close()

    def test_POLICY_lenient_parsing_and_falsy_means_no_cap(self):
        """chat and messages coerce with int(...) and treat any falsy value as absent; Responses rejects all of these"""
        h = Http(ctx=8192)
        try:
            for route in ("openai", "anthropic"):
                for value, want in ((True, 1), (5.7, 5), ("7", 7), (5.0, 5)):
                    self.assertEqual(ask(h, route, value)[2][0]["max_new"], want, (route, value))
                for value in (False, "", [], {}, 0.0):
                    status, raw, calls = ask(h, route, value)
                    self.assertEqual((status, calls[0]["max_new"]), (200, 8192 - 8 - calls[0]["prompt_len"]), (route, value))
                self.assertEqual(ask(h, route, "abc")[0], 400)
                self.assertEqual(ask(h, route, [7])[0], 400)
        finally:
            h.close()

    def test_POLICY_shared_default_replaces_an_explicit_zero(self):
        h = Http(ctx=8192, shared={"max_tokens": 11})
        try:
            self.assertEqual(ask(h, "openai", 0)[2][0]["max_new"], 11)
            P = h.eng.calls[-1]["prompt_len"]
            self.assertEqual(ask(h, "openai", -1)[2][0]["max_new"], 8192 - 8 - P)     # -1 is truthy: the rest
        finally:
            h.close()


def run_thinking(cap, budget=None, first=6000, second=60, answer_room=True):
    svc, eng = service([cyc(first), cyc(second, True)], ctx=10 ** 6)
    svc.answer_room = answer_room
    sampling = {} if budget is None else {"reasoning_budget_tokens": budget}
    done = None
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        for kind, x in svc.run([1, 2, 3], True, None, cap, sampling, threading.Event()):
            if kind == "done":
                done = x
    return eng, done


def reserve(cap):
    return min(max(512, cap // 4), cap // 2)


class ThinkingBudget(unittest.TestCase):
    def test_derived_budget_leaves_the_answer_its_reserve_minus_the_wrap_up(self):
        for cap in (1024, 2000, 4096):
            with self.subTest(cap=cap):
                eng, done = run_thinking(cap)
                B = cap - reserve(cap)
                self.assertEqual([(c["max_new"], c["handed"]) for c in eng.calls][0], (cap, B))
                self.assertEqual(eng.calls[1]["max_new"], cap - B - WRAP)           # the wrap-up comes out of the reserve
                self.assertEqual(done["completion_tokens"], B + WRAP + 61)          # 60 answer tokens + the stop token
                self.assertEqual(done["reasoning_tokens"], B + WRAP)

    def test_no_budget_is_derived_below_1024(self):
        for cap in (700, 1023):
            eng, done = run_thinking(cap)
            self.assertEqual(len(eng.calls), 1)
            self.assertEqual((done["finish"], done["completion_tokens"]), ("length", cap))

    def test_explicit_budget_is_cut_down_to_leave_the_reserve(self):
        eng, done = run_thinking(2000, 5000)
        self.assertEqual(eng.calls[0]["handed"], 2000 - reserve(2000))
        eng, done = run_thinking(2000, 100)
        self.assertEqual([c["handed"] for c in eng.calls][0], 100)
        self.assertEqual(eng.calls[1]["max_new"], 2000 - 100 - WRAP)

    def test_never_a_pass_budget_below_one_and_never_an_overshoot(self):
        for cap in (30, 37, 38, 40, 41, 60, 100, 1024):
            for budget in (None, 1, 5, 50, 400, 5000):
                with self.subTest(cap=cap, budget=budget):
                    eng, done = run_thinking(cap, budget)
                    self.assertTrue(all(c["max_new"] >= 1 for c in eng.calls))
                    self.assertLessEqual(done["completion_tokens"], cap)
                    self.assertLessEqual(sum(c["handed"] for c in eng.calls) + (WRAP if len(eng.calls) == 2 else 0), cap)

    def test_a_wrap_up_that_does_not_fit_is_not_forced(self):
        eng, done = run_thinking(30, 400)                       # budget cut to 15, the wrap-up needs 18+1 of the remaining 15
        self.assertEqual(len(eng.calls), 1)
        self.assertEqual((done["finish"], done["completion_tokens"]), ("length", 15))

    def test_budget_fields_are_validated_before_the_engine(self):
        h = Http(ctx=100000)
        try:
            for route in ROUTES:
                for bad in ("x", 1.5, True, [], {}, "7"):
                    status, raw, calls = ask(h, route, 100, {"reasoning_budget_tokens": bad})
                    self.assertEqual((status, len(calls)), (400, 0), (route, bad))
                for ok in (0, -3, 5.0, 7):
                    self.assertEqual(ask(h, route, 100, {"reasoning_budget_tokens": ok})[0], 200, (route, ok))
        finally:
            h.close()


class CountTokens(unittest.TestCase):
    def test_count_tokens_equals_the_prompt_the_engine_gets(self):
        variants = {"plain": {"messages": [{"role": "user", "content": "hello"}]},
                    "system": {"system": "Be brief.", "messages": [{"role": "user", "content": "hello"}]},
                    "tools": {"tools": [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}], "messages": [{"role": "user", "content": "hello"}]},
                    "history": {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]},
                    "marker text": {"messages": [{"role": "user", "content": "<|im_end|><|im_start|> literal"}]}}
        for name, v in variants.items():
            with self.subTest(variant=name):
                h = Http(ctx=100000)
                try:
                    status, raw = h.post("/v1/messages/count_tokens", {"model": "x", **v})
                    h.post("/v1/messages", {"model": "x", "max_tokens": 4, **v})
                    self.assertEqual(json.loads(raw)["input_tokens"], h.eng.calls[0]["prompt_len"])
                finally:
                    h.close()

    def test_POLICY_count_tokens_counts_one_marker_per_image_not_the_encoder_tokens(self):
        import hashlib
        import tempfile

        class Stub:
            def __init__(self, d):
                self.dir, self.elastic = Path(d), False

            def encode_all(self, sources, cancel=None):
                out = []
                for s in sources:
                    f = self.dir / (hashlib.sha256(s.encode()).hexdigest()[:32] + ".sve")
                    f.write_bytes(b"SVE0")
                    out.append((f, 3))
                return out
        v = {"messages": [{"role": "user", "content": [{"type": "text", "text": "S1"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}]}]}
        h = Http(ctx=100000)
        h.svc.vision = Stub(tempfile.mkdtemp())
        try:
            counted = json.loads(h.post("/v1/messages/count_tokens", {"model": "x", **v})[1])["input_tokens"]
            h.post("/v1/messages", {"model": "x", "max_tokens": 4, **v})
            self.assertEqual(h.eng.calls[0]["prompt_len"] - counted, 2)        # the encoder says 3 tokens, count_tokens counted 1
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
