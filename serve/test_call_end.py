"""A02 (4): the outer end of a wrapped call is searched where the last feed stopped, so the work of a call stays with its
new text however often a value shows `</tool_call>`, `</function>` or a whole example - and the answer is the one the
whole walk gives, in every chunking, with the turn's end settling as before.

    python -m unittest serve.test_call_end -v
"""
import random
import unittest
from unittest import mock

from serve import frontend as F
from serve.frontend import CALL_END, OutputParser, call_end

PE, PS, FE, FS = "</parameter>", "<parameter=", "</function>", "<function="
TOOLS = [{"name": "run", "parameters": {"type": "object", "properties": {"arg": {"type": "string"}, "z": {"type": "string"}}}}]


def reference_call_end(text, settle=False):
    """The whole walk as it was before the resumable one: copies the rest of the text at every step and starts afresh."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith(FS):
        return text.find(CALL_END) if not FS.startswith(s) else -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        if s.startswith(PS):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = F.param_end(text[gt + 1:], settle=settle)
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PE)
        elif s.startswith(FE):
            rest = text[pos + len(FE):]
            s = rest.lstrip()
            if s.startswith(CALL_END):
                return pos + len(FE) + len(rest) - len(s)
            return -1 if CALL_END.startswith(s) else text.find(CALL_END, pos)
        elif not s or PS.startswith(s) or FE.startswith(s):
            return -1
        else:
            return text.find(CALL_END, pos)


ATOMS = [" ", "\n", "\t", FS + "run>", FS, PS + "arg>", PS + "z>", PS, "arg>", PE, FE, CALL_END, "<tool_call>", "x", "ab\n",
         "</param", "</func", "</tool_c", "<paramet", ">", PE + "\n", "\n" + FE, FE + "\n" + CALL_END]


def random_text(rng):
    return "".join(rng.choice(ATOMS) for _ in range(rng.randint(0, 14)))


class CallEndEquivalenceTests(unittest.TestCase):
    def test_same_answer_as_the_whole_copying_walk(self):
        rng = random.Random(1)
        for _ in range(8000):
            text = random_text(rng)
            for settle in (False, True):
                with self.subTest(text=text, settle=settle):
                    self.assertEqual(call_end(text, settle), reference_call_end(text, settle))

    def test_going_on_gives_the_answer_of_a_fresh_walk_at_every_length(self):
        rng = random.Random(2)
        for _ in range(3000):
            text = random_text(rng) + rng.choice(["", CALL_END, FE + "\n" + CALL_END])
            state = {}
            cut = 0
            while cut < len(text):
                cut = min(len(text), cut + rng.randint(1, 9))
                for settle in (False, True):
                    with self.subTest(text=text, cut=cut, settle=settle):
                        self.assertEqual(call_end(text[:cut], settle, state if not settle else dict(state)),
                                         reference_call_end(text[:cut], settle))
                call_end(text[:cut], False, state)                 # keep the state going with the unsettled walk

    def test_a_state_taken_unsettled_serves_the_settled_walk_too(self):
        text = (FS + "run>\n" + PS + "arg>\nsee " + PS + "NAME> here\n" + PE + "\n" + PS + "z>\nt\n" + PE + "\n" + FE + "\n" +
                CALL_END)
        state = {}
        for cut in range(1, len(text) + 1):
            call_end(text[:cut], False, state)
        self.assertEqual(call_end(text, False, state), -1)          # undecided: only the turn's end settles it
        self.assertEqual(call_end(text, True, state), reference_call_end(text, True))
        self.assertGreater(call_end(text, True, state), 0)


class ReferenceParser(OutputParser):
    """The stream parser with the whole walk at every feed (what _wrapped_call_end did)."""

    def _wrapped_call_end(self):
        if not self.recover and self._call_end_checked >= 0:
            start = max(0, self._call_end_checked - len(CALL_END) + 1)
            if self.buf.find(CALL_END, start) < 0:
                self._call_end_checked = len(self.buf)
                return -1
            self._call_end_checked = -1
        return reference_call_end(self.buf, self.settle)


def events_of(parser_class, text, cuts, stream):
    parser = parser_class(thinking=False, tools=TOOLS, stream_tools=stream)
    out, last = [], 0
    for c in list(cuts) + [len(text)]:
        out += parser.feed(text[last:c])
        last = c
    out += parser.finish("stop")
    return [(e.kind, e.text, (e.call.name, e.call.arguments) if e.call else None) for e in out]


def call(value, z="tail"):
    return ("<tool_call>\n<function=run>\n<parameter=arg>\n" + value + "\n</parameter>\n<parameter=z>\n" + z +
            "\n</parameter>\n</function>\n</tool_call>")


class ParserEquivalenceTests(unittest.TestCase):
    def test_events_equal_the_whole_walk_parser_in_every_chunking(self):
        rng = random.Random(3)
        pool = ["a", "\n", " x ", CALL_END, "<tool_call>", FE, FE + "\n" + CALL_END, PE, PS + "q>", "<parameter=q>\nv\n</parameter>",
                "<tool_call>\n<function=f>\n<parameter=q>\nv\n</parameter>\n</function>\n</tool_call>"]
        for n in range(400):
            value = "".join(rng.choice(pool) for _ in range(rng.randint(1, 6)))
            text = call(value) + rng.choice(["", "\ntrailing", "\n" + call("again")])
            if rng.random() < 0.25:
                text = text[:rng.randint(len(text) // 2, len(text))]
            sets = [[]] + [list(range(w, len(text), w)) for w in (1, 5, 8)] + [
                sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(1, 10))))]
            for stream in (False, True):
                for cuts in sets:
                    with self.subTest(n=n, stream=stream, cuts=len(cuts)):
                        self.assertEqual(events_of(OutputParser, text, cuts, stream),
                                         events_of(ReferenceParser, text, cuts, stream))

    def test_two_calls_in_a_turn_do_not_share_a_walk(self):
        text = call("one </tool_call> two", "1") + "\n" + call("three </function> four", "2") + "\n" + call("plain", "3")
        for stream in (False, True):
            for w in (1, 7, len(text)):
                with self.subTest(stream=stream, width=w):
                    got = events_of(OutputParser, text, list(range(w, len(text), w)), stream)
                    self.assertEqual(got, events_of(ReferenceParser, text, list(range(w, len(text), w)), stream))
                    names = [e[2] for e in got if e[0] == "tool_call"]
                    self.assertEqual([a["z"] for _, a in names], ["1", "2", "3"])


class WorkTests(unittest.TestCase):
    """The text the searches may traverse, summed over a whole turn: linear in the turn, not in feeds x length."""

    @staticmethod
    def spans(text, width, stream, resumable):
        total = [0]
        real_find, real_count = F._find, F._count

        def find(t, sub, start=0):
            at = real_find(t, sub, start)
            total[0] += (at if at >= 0 else len(t)) - start      # what the search really passes over
            return at

        def count(t, sub, a, b):
            total[0] += b - a
            return real_count(t, sub, a, b)
        parser_class = OutputParser if resumable else ReferenceParser
        with mock.patch.object(F, "_find", find), mock.patch.object(F, "_count", count):
            parser = parser_class(thinking=False, tools=TOOLS, stream_tools=stream)
            events = []
            for i in range(0, len(text), width):
                events += parser.feed(text[i:i + width])
            events += parser.finish("stop")
        return total[0], [e for e in events if e.kind == "tool_call"]

    def test_an_inner_closing_tag_does_not_make_every_feed_a_whole_walk(self):
        wrapped = "<tool_call>\n<function=a>\n<parameter=p>\nv\n</parameter>\n</function>\n</tool_call>\n"
        body = "line of text in a file\n" * 2800                            # ~64 KB
        for label, value in (("wrapper at the start", wrapped + body), ("literal </tool_call> at the start", "x " + CALL_END + "\n" + body),
                             ("plain", body)):
            text = call(value)
            for stream in (False, True):
                with self.subTest(label=label, stream=stream):
                    resumable, calls = self.spans(text, 16, stream, True)
                    whole, _ = self.spans(text, 16, stream, False)
                    self.assertEqual(len(calls), 1)
                    self.assertLess(resumable, 40 * len(text), (resumable, len(text)))
                    if label != "plain":
                        self.assertGreater(whole, 20 * resumable)         # the reference really is the quadratic walk


if __name__ == "__main__":
    unittest.main()
