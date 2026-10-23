"""Token bytes up to the visible text: Detokenizer, StopMatcher, stop_strings and the real Service.run cut (A06c).

Fast and CPU-only: a fake tokenizer with token_bytes, a scripted fake engine, no model.  Two tests PIN current behaviour that is an open policy
question (marked POLICY): they exist so a change is noticed, not because the behaviour is endorsed."""
import itertools
import random
import threading
import unittest
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Detokenizer, MockEngine, Service, StopMatcher, stop_strings


# ---- an independent UTF-8 oracle (not codecs): what a strict decoder has settled after the bytes so far
def need(lead):
    return 2 if 0xC2 <= lead <= 0xDF else 3 if 0xE0 <= lead <= 0xEF else 4 if 0xF0 <= lead <= 0xF4 else 0


def second_ok(lead, b):
    lo, hi = {0xE0: (0xA0, 0xBF), 0xED: (0x80, 0x9F), 0xF0: (0x90, 0xBF), 0xF4: (0x80, 0x8F)}.get(lead, (0x80, 0xBF))
    return lo <= b <= hi


def pending_tail(b):
    for k in (3, 2, 1):
        if len(b) >= k:
            s = b[len(b) - k:]
            if need(s[0]) > k and (k < 2 or second_ok(s[0], s[1])) and all(0x80 <= x <= 0xBF for x in s[2:]):
                return k
    return 0


def settled(b):
    return b[:len(b) - pending_tail(b)].decode("utf-8", "replace")


class BytesTok:
    def __init__(self, vocab):
        self.vocab = vocab

    def token_bytes(self, t):
        return self.vocab[t]


BYTES = {"ascii": b"hello", "2-byte": "grüße".encode(), "3-byte": "你好".encode(), "emoji": "\U0001f30a\U0001f98a".encode(),
         "combining": "é".encode(), "zwj": "\U0001f469‍\U0001f4bb".encode(), "real U+FFFD": "a�b".encode(), "lone continuation": b"a\x80b",
         "lone lead": b"a\xc3b", "C0 80": b"a\xc0\x80b", "F5": b"a\xf5\x80\x80\x80b", "overlong": b"\xe0\x80\x80x", "truncated 3 then lead": b"\xe4\xb8\xe4\xb8\xad",
         "truncated 4 then ascii": b"\xf0\x9f\x8cx", "F4 90": b"\xf4\x90\x80\x80", "ends in 2": b"ab\xc3", "ends in 3": b"ab\xe4\xb8", "ends in 4": b"ab\xf0\x9f\x8c",
         "ends on invalid": b"ab\xff"}


class DetokenizerBytes(unittest.TestCase):
    def test_every_token_matches_the_byte_level_oracle(self):
        rng = random.Random(11)
        for name, b in BYTES.items():
            parts_list = [[bytes([x]) for x in b], [b]]
            for _ in range(15):
                cuts = sorted(rng.randint(0, len(b)) for _ in range(rng.randint(0, 5)))
                parts, prev = [], 0
                for c in cuts + [len(b)]:
                    parts.append(b[prev:c])
                    prev = c
                parts_list.append(parts)
            for parts in parts_list:
                with self.subTest(case=name, parts=len(parts)):
                    d, got, acc = Detokenizer(BytesTok(parts)), "", b""
                    for i, p in enumerate(parts):
                        acc += p
                        got += d.push(i)
                        if acc[-2:-1] == b"\xed" and 0xA0 <= acc[-1] <= 0xBF:
                            continue                     # CPython holds an invalid surrogate prefix one token longer (text equal after)
                        self.assertEqual(got, settled(acc))
                        self.assertEqual(d.pending(), pending_tail(acc) > 0)

    def test_POLICY_a_character_cut_at_the_end_is_dropped_not_replaced(self):
        for name in ("ends in 2", "ends in 3", "ends in 4"):
            b = BYTES[name]
            d, got = Detokenizer(BytesTok([bytes([x]) for x in b])), ""
            for i in range(len(b)):
                got += d.push(i)
            self.assertEqual(got, "ab")                  # a whole decode would end in U+FFFD; Service.run never flushes the detokenizer
            self.assertTrue(d.pending())


# ---- StopMatcher
def whole(text, stops):
    best = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, s)
    return (text[:best[0]], best[1]) if best else (text, None)


def streamed(text, stops, cuts):
    m, outs, prev = StopMatcher(stops), [], 0
    for c in list(cuts) + [len(text)]:
        outs.append(m.push(text[prev:c]))
        prev = c
    if m.hit is None:
        outs.append(m.flush())
    return "".join(outs), m.hit


def compositions(n):
    for mask in range(1 << max(n - 1, 0)):
        yield [i + 1 for i in range(n - 1) if mask >> i & 1]


def unambiguous(stops):
    for s, t in itertools.permutations(stops, 2):
        if s in t or any(s.endswith(t[:k]) for k in range(1, min(len(s), len(t)))):
            return False
    return True


class StopMatcherContract(unittest.TestCase):
    def test_unambiguous_stops_give_the_whole_text_result_for_every_fragmentation(self):
        words = ["".join(p) for n in (1, 2, 3) for p in itertools.product("ab", repeat=n)]
        texts = ["".join(p) for n in range(0, 6) for p in itertools.product("ab", repeat=n)]
        checked = 0
        for stops in itertools.chain(([w] for w in words), (list(p) for p in itertools.permutations(words, 2))):
            if not unambiguous(stops):
                continue
            for t in texts:
                for cuts in compositions(len(t)):
                    self.assertEqual(streamed(t, stops, cuts), whole(t, stops), (stops, t, cuts))
                    checked += 1
        self.assertGreater(checked, 10000)

    def test_unicode_stops_and_the_held_tail(self):
        stops = ["中文", "é\U0001f600"]
        for t in ("x 中文 y", "café é\U0001f600!", "ends 中", "é"):
            for cuts in compositions(len(t)):
                self.assertEqual(streamed(t, stops, cuts), whole(t, stops), (t, cuts))
        m = StopMatcher(["END"])
        self.assertEqual((m.push("ab EN"), m.held), ("ab ", "EN"))
        self.assertEqual((m.flush(), m.flush()), ("EN", ""))

    def test_after_a_hit_nothing_more_comes(self):
        m = StopMatcher(["END"])
        self.assertEqual((m.push("xxEN"), m.push("Dyy"), m.push("more"), m.flush(), m.hit, m.held), ("xx", "", "", "", "END", ""))
        self.assertEqual(StopMatcher(["END"]).push("D"), "D")           # instances share nothing

    def test_POLICY_overlapping_stops_depend_on_the_fragmentation(self):
        """a stop that starts later can complete first: whole text says 'baa', streaming 'b|a|a' says 'a'"""
        self.assertEqual(streamed("baa", ["a", "baa"], []), ("", "baa"))
        self.assertEqual(streamed("baa", ["a", "baa"], [1, 2]), ("b", "a"))


class StopStringsInput(unittest.TestCase):
    def test_forms(self):
        ok = {None: [], "x": ["x"], "": [], ("a", "", "b"): ["a", "b"], ("a", "b", "c", "d"): ["a", "b", "c", "d"]}
        for stop, want in ok.items():
            self.assertEqual(stop_strings({"stop": list(stop) if isinstance(stop, tuple) else stop}), want)
        for bad in (5, True, [1], [None], {"a": 1}, ["a", "b", "c", "d", "e"]):
            with self.assertRaises(ValueError):
                stop_strings({"stop": bad})
        self.assertEqual(stop_strings({"stop_sequences": list("abcdef")}), list("abcdef"))   # no count limit (policy open)
        for bad in ("x", [1], {"a": 1}):
            with self.assertRaises(ValueError):
                stop_strings({"stop_sequences": bad})
        self.assertEqual(stop_strings({"stop": "x", "stop_sequences": ["y"]}), ["x"])
        self.assertEqual(stop_strings(None), [])


# ---- the real Service.run with a scripted engine
class Tok(ByteTokenizer):
    def __init__(self):
        self.extra = []

    def token_bytes(self, t):
        return self.extra[t - 512] if t >= 512 else self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])

    def decode(self, ids, errors="replace"):
        return b"".join(self.token_bytes(t) for t in ids).decode("utf-8", errors)

    def ids(self, text, cuts):
        b, out, prev = text.encode(), [], 0
        for c in sorted(set(cuts)) + [len(b)]:
            if c > prev:
                piece = b[prev:c]
                if len(piece) == 1:
                    out.append(piece[0])
                else:
                    if piece not in self.extra:
                        self.extra.append(piece)
                    out.append(512 + self.extra.index(piece))
                prev = c
        return out


class Script(MockEngine):
    def __init__(self, tok, ids, cancel_after=None, cancel=None):
        super().__init__(tok, "x", 8192)
        self.script_ids, self.handed, self.closed, self.cancel_after, self.cancel_event = ids, 0, False, cancel_after, cancel

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.handed, self.closed = 0, False
        try:
            for t in (self.script_ids + [257])[:max_new]:
                if self.cancel_after is not None and self.handed == self.cancel_after:
                    self.cancel_event.set()
                if cancel.is_set():
                    return
                self.handed += 1
                yield t
        finally:
            self.closed = True


TOOLS = [{"name": "f", "parameters": {"type": "object", "properties": {"s": {"type": "string"}}}}]
CALL = "<tool_call>\n<function=f>\n<parameter=s>\nhello\n</parameter>\n</function>\n</tool_call>"


def drive(text, cuts, sampling, thinking=False, tools=None, max_new=400, cancel_after=None):
    tok = Tok()
    cancel = threading.Event()
    engine = Script(tok, tok.ids(text, cuts), cancel_after, cancel)
    server.Service.start_telemetry = lambda self: None
    svc = Service(engine, tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
    evs, done = [], None
    for kind, x in svc.run([1, 2, 3], thinking, tools, max_new, sampling, cancel):
        if kind == "event":
            evs.append(x)
        elif kind == "done":
            done = x
    drive.events = evs                                                  # the event order, for tests that need it
    joined = lambda k: "".join(e.text for e in evs if e.kind == k)      # noqa: E731
    return joined("content"), joined("reasoning"), done, [e.call.arguments for e in evs if e.kind == "tool_call"], engine


class ServiceRunCut(unittest.TestCase):
    def tokenisations(self, text):
        n = len(text.encode())
        yield "every byte", list(range(1, n))
        yield "one token", []
        rng = random.Random(7)
        for i in range(8):
            yield f"random {i}", rng.sample(range(1, n), min(n - 1, rng.randint(1, 6)))

    def check(self, text, sampling, want_content, want_stop, **kw):
        for label, cuts in self.tokenisations(text):
            with self.subTest(tokens=label):
                content, reasoning, done, calls, engine = drive(text, cuts, sampling, **kw)
                self.assertEqual(content, want_content)
                self.assertEqual(done.get("stop_sequence"), want_stop)
                self.assertEqual(done["finish"], "stop")
                self.assertTrue(engine.closed)
                yield content, reasoning, done, calls, engine

    def test_content_is_cut_at_the_stop_and_the_engine_closed_at_its_token(self):
        for _, _, _, _, engine in self.check("café 中文 tail END more", {"stop": ["END"]}, "café 中文 tail ", "END"):
            self.assertLessEqual(engine.handed, len("café 中文 tail END".encode()))

    def test_unicode_stop_split_inside_a_character(self):
        list(self.check("x 中文 y", {"stop": ["中文"]}, "x ", "中文"))

    def test_a_held_prefix_is_flushed_at_the_end_when_no_stop_came(self):
        list(self.check("plain text ends with EN", {"stop": ["END"]}, "plain text ends with EN", None))

    def test_reasoning_is_never_cut_and_the_boundary_is_not_a_stop_boundary(self):
        for _, reasoning, *_ in self.check("think END think</think>\n\nanswer END tail", {"stop": ["END"]}, "answer ", "END", thinking=True):
            self.assertEqual(reasoning, "think END think")
        for _, reasoning, *_ in self.check("think EN</think>\n\nD answer", {"stop": ["END"]}, "D answer", None, thinking=True):
            self.assertEqual(reasoning, "think EN")

    def test_a_held_prefix_is_flushed_before_a_tool_call_and_arguments_are_not_cut(self):
        for content, _, _, calls, _ in self.check("ab" + CALL, {"stop": ["abc"]}, "ab", None, tools=TOOLS):
            self.assertEqual(calls, [{"s": "hello"}])
            kinds = [e.kind for e in drive.events]
            self.assertLess(kinds.index("content"), kinds.index("tool_start"))      # flushed BEFORE the call, not after it
        for _, _, _, calls, _ in self.check(CALL.replace("hello", "xx END yy"), {"stop": ["END"]}, "", None, tools=TOOLS):
            self.assertEqual(calls, [{"s": "xx END yy"}])
        for _, _, _, calls, _ in self.check("pre END mid\n" + CALL, {"stop": ["END"]}, "pre ", "END", tools=TOOLS):
            self.assertEqual(calls, [])                   # the call after the stop is dropped

    def test_length_inside_a_stop_prefix_flushes_it_and_cancel_still_shows_it(self):
        content, _, done, _, _ = drive("abc END", list(range(1, 7)), {"stop": ["END"]}, max_new=len(b"abc EN"))
        self.assertEqual((content, done["finish"]), ("abc EN", "length"))
        content, _, done, _, _ = drive("abc EN more", list(range(1, 11)), {"stop": ["END"]}, cancel_after=6)
        self.assertEqual((content, done["finish"]), ("abc EN", "cancel"))


if __name__ == "__main__":
    unittest.main()
