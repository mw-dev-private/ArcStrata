"""Handler._captured (the monitor's bounded answer/reasoning record), run as the real method; no HTTP, no model."""
import threading
import unittest
from types import SimpleNamespace

from serve import server

CAP = 262144


def handler_class():
    return server.make_handler(SimpleNamespace(status_lock=threading.RLock()))


def blank():
    return {"output": "", "reasoning": "", "output_truncated": False, "reasoning_truncated": False}


def openai(content="", reasoning="", usage=None, timings=None):
    item = {"choices": [{"delta": {"content": content, "reasoning_content": reasoning}}]}
    if usage:
        item["usage"] = usage
    if timings:
        item["timings"] = timings
    return item


def responses(kind, delta):
    return {"type": kind, "delta": delta}


def anthropic(text="", thinking="", usage=None):
    event = {"delta": {"text": text, "thinking": thinking}}
    if usage:
        event["usage"] = usage
    return ("content_block_delta", event)


class Source:
    """An iterator that records whether it was closed (the finally in _captured must close it)."""
    def __init__(self, items, fail_after=None):
        self.items, self.fail_after, self.closed, self.taken = list(items), fail_after, False, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.fail_after is not None and self.taken == self.fail_after:
            raise RuntimeError("source failed")
        if self.taken == len(self.items):
            raise StopIteration
        self.taken += 1
        return self.items[self.taken - 1]

    def close(self):
        self.closed = True


def run(api, items, record=None, **kw):
    record = blank() if record is None else record
    fake = SimpleNamespace(record=record)
    source = Source(items, **kw)
    out = list(handler_class()._captured(fake, source, api))
    return record, out, source


class Captured(unittest.TestCase):
    def test_boundary_flag_only_when_really_exceeded(self):
        for n, flag in ((0, False), (1, False), (CAP - 1, False), (CAP, False), (CAP + 1, True), (3 * CAP, True)):
            with self.subTest(n=n):
                record, _, _ = run("openai", [openai("a" * n)])
                self.assertEqual(record["output"], "a" * min(n, CAP))
                self.assertIs(record["output_truncated"], flag)

    def test_boundary_across_fragments(self):
        pieces = ["b" * 100000, "c" * 100000, "d" * 62144, "e"]          # the last fragment is char CAP+1
        record, _, _ = run("openai", [openai(p) for p in pieces[:3]])
        self.assertEqual((len(record["output"]), record["output_truncated"]), (CAP, False))
        record, _, _ = run("openai", [openai(p) for p in pieces])
        self.assertEqual(record["output"], "".join(pieces)[:CAP])
        self.assertTrue(record["output_truncated"])

    def test_fields_are_independent(self):
        items = [openai(content="a" * (CAP + 5)), openai(reasoning="r1"), openai(content="late", reasoning="r2")]
        record, _, _ = run("openai", items)
        self.assertEqual((record["output"], record["output_truncated"]), ("a" * CAP, True))
        self.assertEqual((record["reasoning"], record["reasoning_truncated"]), ("r1r2", False))
        record, _, _ = run("openai", [openai(reasoning="r" * (CAP + 1)), openai(content="x"), openai(reasoning="more")])
        self.assertEqual((record["output"], record["output_truncated"]), ("x", False))
        self.assertEqual((record["reasoning"], record["reasoning_truncated"]), ("r" * CAP, True))

    def test_usage_and_timings_after_the_text_boundary(self):
        items = [openai("a" * (CAP + 1)), openai("zz"), openai(usage={"completion_tokens": 9}, timings={"t": 1})]
        record, out, _ = run("openai", items)
        self.assertEqual(record["usage"], {"completion_tokens": 9})
        self.assertEqual(record["timings"], {"t": 1})
        self.assertEqual(len(out), 3)
        record, _, _ = run("responses", [responses("response.output_text.delta", "a" * (CAP + 1)),
                                         {"type": "response.completed", "response": {"usage": {"output_tokens": 5}}}])
        self.assertEqual(record["usage"], {"output_tokens": 5})
        record, _, _ = run("anthropic", [anthropic("a" * (CAP + 1)), anthropic(usage={"output_tokens": 3})])
        self.assertEqual(record["usage"], {"output_tokens": 3})

    def test_events_pass_through_unchanged_in_order_with_none(self):
        items = [openai("a"), None, openai("b" * (CAP + 1)), None, openai("c")]
        record, out, source = run("openai", items)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(o is i for o, i in zip(out, items)))          # identity: not copied, not mutated
        self.assertTrue(source.closed)

    def test_three_dialects_and_unicode(self):
        wide = "€\U0001f600" * ((CAP // 2) + 1)                         # more than CAP characters, many bytes each
        for api, mk_c, mk_r in (("openai", lambda t: openai(content=t), lambda t: openai(reasoning=t)),
                                ("responses", lambda t: responses("response.output_text.delta", t), lambda t: responses("response.reasoning_text.delta", t)),
                                ("anthropic", lambda t: anthropic(text=t), lambda t: anthropic(thinking=t))):
            with self.subTest(api=api):
                record, _, _ = run(api, [mk_c(wide), mk_r("r"), mk_c("tail"), mk_r("s")])
                self.assertEqual(record["output"], wide[:CAP])
                self.assertTrue(record["output_truncated"])
                self.assertEqual((record["reasoning"], record["reasoning_truncated"]), ("rs", False))

    def test_errors_and_close(self):
        source_items = [openai("a" * (CAP + 1)), openai("b")]
        fake = SimpleNamespace(record=blank())
        source = Source(source_items, fail_after=1)
        gen = handler_class()._captured(fake, source, "openai")
        next(gen)
        with self.assertRaises(RuntimeError):
            next(gen)
        self.assertTrue(source.closed)
        self.assertEqual(fake.record["output"], "a" * CAP)
        source = Source(source_items)                                       # the consumer stops early
        gen = handler_class()._captured(SimpleNamespace(record=blank()), source, "openai")
        next(gen)
        gen.close()
        self.assertTrue(source.closed)

    def test_truncated_field_is_not_copied_again(self):
        """After the flag, the kept text is left alone: a str subclass whose __add__ counts calls sees none."""
        calls = []

        class Counting(str):
            def __add__(self, other):
                calls.append(other[:5])
                return Counting(str.__add__(self, other))
        record = blank()
        record["output"], record["output_truncated"] = Counting("k" * CAP), True
        run("openai", [openai("x" * 1000) for _ in range(50)], record=record)
        self.assertEqual(calls, [])
        self.assertEqual(len(record["output"]), CAP)
        record = blank()
        record["output"] = Counting("k" * 10)
        run("openai", [openai("x")], record=record)                           # positive control: below the cap it does add
        self.assertEqual(calls, ["x"])

    def test_flag_without_a_full_prefix_still_appends(self):
        record = blank()
        record["output_truncated"] = True                                     # inconsistent record: not skipped
        run("openai", [openai("abc")], record=record)
        self.assertEqual(record["output"], "abc")

    def test_monitor_off_is_a_pass_through(self):
        items = [openai("a")]
        fake = SimpleNamespace(record=None)
        self.assertIs(handler_class()._capture(fake, items, "openai"), items)


if __name__ == "__main__":
    unittest.main()
