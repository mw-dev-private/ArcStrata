"""A streamed call whose `</function>` was read but whose outer `</tool_call>` never came: the final event carries the call's real
arguments (the evaluation a closed call gets), under the id already announced - and run_with_mcp hands them to the hub."""
import contextlib
import io
import itertools
import json
import threading
import unittest
import unittest.mock

from serve import frontend, server
from serve.frontend import Event, OutputParser

TOOLS = [{"name": "f", "parameters": {"type": "object", "properties": {
    "arr": {"type": "array"}, "obj": {"type": "object"}, "s": {"type": "string"}, "t": {"type": "string"}, "n": {"type": "integer"}}}}]
P = lambda name, value: f"<parameter={name}>\n{value}\n</parameter>\n"        # noqa: E731

CASES = {
    "raw array": (P("arr", "[1, 2, 3]"), {"arr": [1, 2, 3]}),
    "raw object": (P("obj", '{"a": [1, {"b": null}]}'), {"obj": {"a": [1, {"b": None}]}}),
    "string": (P("s", "one\ntwo"), {"s": "one\ntwo"}),
    "string with a literal closer": (P("s", "x </tool_call> y"), {"s": "x </tool_call> y"}),
    "str then raw then str": (P("s", "a") + P("arr", "[9]") + P("t", "c"), {"s": "a", "arr": [9], "t": "c"}),
    "raw integer": (P("n", "7"), {"n": 7}),
    "nested whole example in a raw value": (P("arr", '["<parameter=s>\\nv\\n</parameter>"]'), {"arr": ["<parameter=s>\nv\n</parameter>"]}),
    "no parameters": ("", {}),
}


def run(params, closed, width, recover=False, reason="stop"):
    text = "<tool_call><function=f>\n" + params + "</function>" + ("</tool_call>" if closed else "")
    parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True, recover=recover)
    events, per_feed = [], []
    for i in range(0, len(text), width):
        new = parser.feed(text[i:i + width])
        events += new
        per_feed.append(new)
    tail = parser.finish(reason)
    return parser, events, tail


def shape(events):
    return [(e.kind, e.text, e.call.name if e.call else None) for e in events]


class FinalArguments(unittest.TestCase):
    def test_final_event_carries_the_calls_real_arguments_and_announced_id(self):
        for (name, (params, want)), width in itertools.product(CASES.items(), (1, 5, 17)):
            for recover in (False, True):
                with self.subTest(case=name, width=width, recover=recover):
                    _, events, tail = run(params, False, width, recover)
                    starts = [e for e in events if e.kind == "tool_start"]
                    finals = [e for e in events + tail if e.kind == "tool_call"]
                    self.assertEqual(len(starts), 1)
                    self.assertEqual(len(finals), 1)
                    self.assertEqual(finals[0].call.arguments, want)
                    self.assertEqual(finals[0].call.name, "f")
                    self.assertEqual(finals[0].call.id, starts[0].call.id)
                    streamed = "".join(e.text for e in events + tail if e.kind == "tool_args")
                    self.assertEqual(json.loads(streamed), want)

    def test_same_as_the_closed_call_and_nothing_else_differs(self):
        for (name, (params, want)), width in itertools.product(CASES.items(), (1, 5, 17)):
            with self.subTest(case=name, width=width):
                _, open_events, open_tail = run(params, False, width)
                _, closed_events, closed_tail = run(params, True, width)
                a, b = open_events + open_tail, closed_events + closed_tail
                self.assertEqual(shape(a), shape(b))
                self.assertEqual([e.call.arguments for e in a if e.kind == "tool_call"], [e.call.arguments for e in b if e.kind == "tool_call"])

    def test_earlier_events_are_not_changed_afterwards(self):
        _, events, tail = run(CASES["raw array"][0], False, 5)
        start = next(e for e in events if e.kind == "tool_start")
        final = next(e for e in tail if e.kind == "tool_call")
        self.assertEqual(start.call.arguments, {})                      # the announced object stays as it went out
        self.assertIsNot(start.call, final.call)
        self.assertEqual(final.call.arguments, {"arr": [1, 2, 3]})

    def test_incomplete_calls_get_no_final_event(self):
        for label, text in (("no </function>", "<tool_call><function=f>\n" + P("arr", "[1]")),
                            ("open parameter", "<tool_call><function=f>\n<parameter=arr>\n[1"),
                            ("string cut", "<tool_call><function=f>\n<parameter=s>\nabc")):
            for reason in (None, "stop", "length", "cancel"):
                with self.subTest(case=label, reason=reason):
                    parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
                    events = []
                    for i in range(0, len(text), 5):
                        events += parser.feed(text[i:i + 5])
                    events += parser.finish(reason)
                    self.assertEqual([e for e in events if e.kind == "tool_call"], [])

    def test_a_second_call_and_a_second_finish_carry_nothing_over(self):
        parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
        first = "<tool_call><function=f>\n" + P("arr", "[1]") + "</function>"
        out = parser.feed(first) + parser.finish("stop")
        self.assertEqual([e.call.arguments for e in out if e.kind == "tool_call"], [{"arr": [1]}])
        self.assertEqual(parser.finish("stop"), [])                     # nothing is announced twice
        second = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
        text = first + "</tool_call>\n<tool_call><function=f>\n" + P("s", "x") + "</function>"
        events = second.feed(text) + second.finish("stop")
        self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [{"arr": [1]}, {"s": "x"}])
        ids = [e.call.id for e in events if e.kind in ("tool_start",)]
        self.assertEqual([e.call.id for e in events if e.kind == "tool_call"], ids)

    def test_a_body_that_does_not_parse_keeps_the_announced_call(self):
        parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
        parser.feed("<tool_call><function=f>\n" + P("arr", "[1]") + "</function>")
        with unittest.mock.patch.object(frontend, "parse_tool_call", side_effect=ValueError("x")):
            tail = parser.finish("stop")
        self.assertEqual([e.call.arguments for e in tail if e.kind == "tool_call"], [{}])


class FakeService:
    def __init__(self, events, finish="stop"):
        self.events, self.finish, self.rounds = events, finish, 0

    def run(self, *args, **kwargs):
        self.rounds += 1
        if self.rounds == 1:
            for ev in self.events:
                yield "event", ev
        yield "done", {"finish": self.finish, "completion_tokens": 1, "prompt_tokens": 1}

    def prepare(self, *args, **kwargs):
        return [1], False, 10


class FakeHub:
    settings = {"max_rounds": 1}

    def __init__(self):
        self.seen = []

    def routes(self):
        return {}

    def call(self, name, args, cancel):
        self.seen.append((name, args))
        return {"ok": True, "text": "fake result", "chars": 11, "ms": 0, "truncated": False}


def through_mcp(params, closed, finish="stop", cancelled=False):
    _, events, tail = run(params, closed, 5, reason=finish)
    hub, cancel = FakeHub(), threading.Event()
    if cancelled:
        cancel.set()
    with contextlib.redirect_stdout(io.StringIO()):
        out = list(server.run_with_mcp(FakeService(events + tail, finish), hub, [], TOOLS, {}, [1], False, 10, 10, {}, cancel, {"f"}))
    return hub.seen, out


class ThroughMcp(unittest.TestCase):
    def test_missing_outer_closer_reaches_the_hub_with_its_arguments(self):
        for name in ("raw array", "string", "str then raw then str"):
            params, want = CASES[name]
            with self.subTest(case=name):
                self.assertEqual(through_mcp(params, False)[0], [("f", want)])
                self.assertEqual(through_mcp(params, True)[0], [("f", want)])      # the closed call: unchanged

    def test_length_and_cancel_still_run_nothing(self):
        params, _ = CASES["raw array"]
        for closed in (False, True):
            with self.subTest(closed=closed, finish="length"):
                self.assertEqual(through_mcp(params, closed, finish="length")[0], [])
            with self.subTest(closed=closed, finish="cancel"):
                self.assertEqual(through_mcp(params, closed, finish="cancel")[0], [])
            with self.subTest(closed=closed, cancel_set=True):
                self.assertEqual(through_mcp(params, closed, cancelled=True)[0], [])

    def test_an_incomplete_function_runs_nothing(self):
        hub, cancel = FakeHub(), threading.Event()
        parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
        text = "<tool_call><function=f>\n" + P("arr", "[1]")
        events = []
        for i in range(0, len(text), 5):
            events += parser.feed(text[i:i + 5])
        events += parser.finish("stop")
        with contextlib.redirect_stdout(io.StringIO()):
            list(server.run_with_mcp(FakeService(events), hub, [], TOOLS, {}, [1], False, 10, 10, {}, cancel, {"f"}))
        self.assertEqual(hub.seen, [])


class Emitters(unittest.TestCase):
    """The API emitters skip the final event of an already streamed id: a missing outer closer changes nothing for them."""

    def chunks(self, closed):
        _, events, tail = run(CASES["raw array"][0], closed, 5)
        evs = events + tail

        class Svc:
            engine = type("E", (), {"batch": 0, "progress": None})()
            def model_for(self, req): return "m"
            def run(self, *a, **k):
                for ev in evs:
                    yield "event", ev
                yield "done", {"finish": "stop", "completion_tokens": 3, "prompt_tokens": 1}
        out = list(server.openai_chunks(Svc(), {}, [1, 2], True, None, 100, threading.Event()))
        for c in out:
            c["id"], c["created"] = "x", 0
            for tc in (c["choices"][0]["delta"].get("tool_calls") or []):
                tc.pop("id", None)
        return out

    def test_openai_chunks_are_the_same_with_and_without_the_outer_closer(self):
        self.assertEqual(self.chunks(False), self.chunks(True))


if __name__ == "__main__":
    unittest.main()
