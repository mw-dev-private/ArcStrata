"""A02 (2): a string value may carry whole `<parameter=>` / `<function=>` / `<tool_call>` examples (a file that
documents the call format).  The value ends where no example is open any more, in every chunking.  Where no
`<parameter=` stands inside a value the end is the plain rule it always was; the unbalanced boundary
(`</parameter>` + `<parameter=` with nothing open) stays what the format cannot tell apart from two arguments."""
import json
import random
import re
import unittest
from unittest import mock

from serve import frontend as F
from serve.frontend import OutputParser, param_depth, param_end, parse_tool_call

PE, PS, FE = "</parameter>", "<parameter=", "</function>"
TOOLS = [{"name": "run", "parameters": {"type": "object", "properties": {"arg": {"type": "string"}, "z": {"type": "string"}}}}]


def old_param_end(text, final=False):
    """The rule before this change (inline copy)."""
    at = text.find(PE)
    while at >= 0:
        after = text[at + len(PE):].lstrip()
        if after.startswith((PS, FE)):
            return at
        if not after or PS.startswith(after) or FE.startswith(after):
            return at if final else -2
        at = text.find(PE, at + 1)
    return -1


def oracle(text, final=False, settle=False, depth=0):
    """Independent statement of the rule: a `</parameter>` ends the value when what follows it is the next
    `<parameter=` / `</function>` and no `<parameter=` opened before it is still open (replayed from the start)."""
    first = -1
    for m in re.finditer(re.escape(PE), text):
        d = param_depth(text[:m.start()], depth)
        after = text[m.end():].lstrip()
        if after.startswith((PS, FE)):
            kind = "end"
        elif not after or PS.startswith(after) or FE.startswith(after):
            if not final:
                return -2
            kind = "end"
        else:
            kind = "text"
        if kind == "end":
            if d == 0:
                return m.start()
            if first < 0:
                first = m.start()
    if first >= 0:
        return first if (settle or final) else -2
    return -1


def call(value, name="run", z="tail"):
    return ("<tool_call>\n<function=" + name + ">\n<parameter=arg>\n" + value + "\n</parameter>\n<parameter=z>\n" +
            z + "\n</parameter>\n</function>\n</tool_call>")


def run(text, cuts, stream, reason="stop"):
    parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream)
    events, last = [], 0
    for c in list(cuts) + [len(text)]:
        events += parser.feed(text[last:c])
        last = c
    events += parser.finish(reason)
    calls = [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"]
    content = "".join(e.text for e in events if e.kind == "content")
    streamed = "".join(e.text for e in events if e.kind == "tool_args")
    return calls, content, streamed, events


def chunkings(n, rng):
    yield []
    for w in (1, 3, 7, 8):
        yield list(range(w, n, w))
    for _ in range(3):
        yield sorted(rng.sample(range(1, n), min(n - 1, rng.randint(1, 12)))) if n > 1 else []


class ParamEndRuleTests(unittest.TestCase):
    def test_plain_rule_where_no_opener_stands_inside_the_value(self):
        rng = random.Random(1)
        atoms = [PE, FE, "\n", " ", "x", "</param", "</func", "<parameter", "<param", "</parameter> ", "text"]
        for _ in range(4000):
            text = "".join(rng.choice(atoms) for _ in range(rng.randint(0, 9)))
            if PS in text:
                continue
            for final in (False, True):
                with self.subTest(text=text, final=final):
                    self.assertEqual(param_end(text, final), old_param_end(text, final))
                    self.assertEqual(param_end(text, final, settle=True), old_param_end(text, final))

    def test_agrees_with_the_independent_statement_of_the_rule(self):
        rng = random.Random(2)
        atoms = [PE, FE, PS, "\n", " ", "x", "</param", "<param", "<parameter=q>", "</parameter>\n", "t"]
        for _ in range(6000):
            text = "".join(rng.choice(atoms) for _ in range(rng.randint(0, 12)))
            depth = rng.choice([0, 0, 0, 1, 2])
            for final in (False, True):
                for settle in (False, True):
                    with self.subTest(text=text, final=final, settle=settle, depth=depth):
                        self.assertEqual(param_end(text, final, settle, depth), oracle(text, final, settle, depth))

    def test_whitespace_run_is_what_lstrip_strips(self):
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            ch = chr(cp)
            self.assertEqual(F._WS_RUN.match(ch).end() == 1, ch.isspace(), hex(cp))

    def test_named_cases(self):
        cases = [
            ("v\n</parameter>\n</function>\nend\n</parameter>\n<parameter=z>", None),    # no opener: first candidate
            ("<parameter=p>\nv\n</parameter>\n</function>\nend\n</parameter>\n<parameter=z>\n", "end\n"),
        ]
        text, _ = cases[0]
        self.assertEqual(param_end(text), text.index(PE))
        text = cases[1][0]
        self.assertEqual(param_end(text), text.rindex(PE, 0, text.index("<parameter=z>")))
        # an unclosed mention: only a later end can settle it - undecided, then the old rule's end at the turn's end
        mention = "see <parameter=NAME> here\n</parameter>\n<parameter=z>\nt\n</parameter>\n</function>"
        self.assertEqual(param_end(mention), -2)
        self.assertEqual(param_end(mention, settle=True), mention.index(PE))
        self.assertEqual(param_end(mention, final=True), mention.index(PE))

    def test_depth_carried_over_handed_out_text(self):
        self.assertEqual(param_depth("a <parameter=p> b"), 1)
        self.assertEqual(param_depth("a <parameter=p> b </parameter> c"), 0)
        self.assertEqual(param_depth("</parameter> a"), 0)                      # nothing open: ignored
        self.assertEqual(param_depth("<parameter=a><parameter=b></parameter>"), 1)
        self.assertEqual(param_depth("<parameter=a>", 2), 3)


class EmbeddedExampleTests(unittest.TestCase):
    VALUES = {
        "function example": "doc:\n<function=a>\n<parameter=p>\nv\n</parameter>\n</function>\nend",
        "wrapped example": "doc:\n<tool_call>\n<function=a>\n<parameter=p>\nv\n</parameter>\n</function>\n</tool_call>\nend",
        "two parameter examples": "<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>",
        "nested examples": "<parameter=a>\n<parameter=b>\n1\n</parameter>\n</parameter>\n",
        "example at the end": "x\n<parameter=p>\nv\n</parameter>",
        "example first": "<parameter=p>\nv\n</parameter>\nrest of the file",
        "two wrapped examples": "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>",
        "plain": "a\nb\nc",
        "lone terminator": "x\n</parameter>",
        "#210 balanced fragment": "use <parameter=p>\nv\n</parameter> inside text",
        "unclosed mention": "syntax is <parameter=NAME> followed by a value",
        "unclosed mention, closer-like prose": "a <parameter=NAME> b\n</parameter>\nand prose",
        "closing tags only": "x\n</function>\n</tool_call>\ny",
    }

    def test_every_value_comes_back_exact_in_every_chunking(self):
        rng = random.Random(3)
        for label, value in self.VALUES.items():
            text = call(value)
            for stream in (False, True):
                for cuts in chunkings(len(text), rng):
                    with self.subTest(case=label, stream=stream, cuts=len(cuts)):
                        calls, content, streamed, _ = run(text, cuts, stream)
                        self.assertEqual(calls, [("run", {"arg": value, "z": "tail"})])
                        self.assertEqual(content.strip(), "")
                        if stream:
                            self.assertEqual(json.loads(streamed), calls[0][1])

    def test_direct_parser_agrees(self):
        for label, value in self.VALUES.items():
            body = call(value)[len("<tool_call>\n"):-len("\n</tool_call>")]
            with self.subTest(case=label):
                self.assertEqual(parse_tool_call(body, TOOLS[0]).arguments, {"arg": value, "z": "tail"})

    def test_the_unbalanced_boundary_stays_ambiguous(self):
        """Nothing is open when `</parameter>` + `<parameter=` appears: it reads as two arguments, as before."""
        value = "hello\n</parameter>\n<parameter=other>\nINJECTED"
        calls = run(call(value), [], False)[0]
        self.assertEqual(calls, [("run", {"arg": "hello", "other": "INJECTED", "z": "tail"})])

    def test_several_calls_in_a_turn_with_examples_and_mentions(self):
        text = ("first: " + call(self.VALUES["unclosed mention"], z="1") + "\nthen\n" +
                call(self.VALUES["wrapped example"], z="2") + "\nand " + call("plain", z="3") + "\nbye")
        want = [("run", {"arg": self.VALUES["unclosed mention"], "z": "1"}),
                ("run", {"arg": self.VALUES["wrapped example"], "z": "2"}),
                ("run", {"arg": "plain", "z": "3"})]
        rng = random.Random(4)
        for stream in (False, True):
            for cuts in chunkings(len(text), rng):
                with self.subTest(stream=stream, cuts=len(cuts)):
                    calls, content, _, _ = run(text, cuts, stream)
                    self.assertEqual(calls, want)
                    self.assertEqual("".join(content.split()), "first:thenandbye")

    def test_unterminated_calls_are_no_calls(self):
        """A turn cut inside a value never yields a call, however the value reads, in every chunking."""
        for label in ("function example", "unclosed mention", "plain", "wrapped example"):
            full = call(self.VALUES[label])
            text = full[:full.index("<parameter=z>") + len("<parameter=z>\nta")]      # cut inside the last value
            rng = random.Random(7)
            for stream in (False, True):
                for cuts in chunkings(len(text), rng):
                    with self.subTest(case=label, stream=stream, cuts=len(cuts)):
                        calls, content, _, events = run(text, cuts, stream)
                        self.assertEqual(calls, [])
                        self.assertFalse(any(e.kind == "tool_call" for e in events))
                        if not stream:
                            self.assertTrue(content.startswith("<tool_call>"))
                            self.assertEqual(content, text)

    def test_chunking_never_changes_the_outcome(self):
        """Random turns made of examples, fragments and unterminated tails: every chunking agrees with the whole text."""
        rng = random.Random(6)
        pool = ["a", "\n", " x ", PS + "q>", PE, PE + "\n", FE, "<function=f>", "<tool_call>", "</tool_call>",
                "<parameter=q>\nv\n</parameter>", "<tool_call>\n<function=f>\n<parameter=q>\nv\n</parameter>\n</function>\n</tool_call>"]
        for n in range(500):
            value = "".join(rng.choice(pool) for _ in range(rng.randint(1, 7)))
            text = call(value) + rng.choice(["", "\ntrailing", "\n" + call("again")])
            if rng.random() < 0.25:
                text = text[:rng.randint(len(text) // 2, len(text))]
            for stream in (False, True):
                whole = run(text, [], stream)[:2]
                for cuts in chunkings(len(text), rng):
                    with self.subTest(n=n, stream=stream, cuts=len(cuts)):
                        got = run(text, cuts, stream)
                        self.assertEqual(got[:2], whole)

    def test_a_held_value_is_not_rescanned_for_every_feed(self):
        """The scanner holds a long value back until its end is known; the work stays with the new text. A whole
        rescan per feed would weigh every closer again on each of the ~4000 feeds."""
        text = call("<function=a>\n<parameter=p>\nv\n</parameter>\n</function>\n" + "line of text in a file\n" * 3000)
        weighed, events = [], []
        real = F._follows
        with mock.patch.object(F, "_follows", lambda t, j: weighed.append(j) or real(t, j)):
            parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
            feeds = 0
            for i in range(0, len(text), 16):
                events += parser.feed(text[i:i + 16])
                feeds += 1
            events += parser.finish("stop")
        self.assertGreater(feeds, 4000)
        self.assertLess(len(weighed), 40)
        self.assertEqual([e.call.name for e in events if e.kind == "tool_call"], ["run"])

    def test_inputs_are_not_mutated(self):
        tools = json.loads(json.dumps(TOOLS))
        parser = OutputParser(thinking=False, tools=tools, stream_tools=True)
        parser.feed(call(self.VALUES["wrapped example"]))
        parser.finish("stop")
        self.assertEqual(tools, TOOLS)


if __name__ == "__main__":
    unittest.main()
