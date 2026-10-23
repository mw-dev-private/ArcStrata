"""anthropic_collect regressions (audit 2026-10-21): the block-buffer accumulation must return dictionaries
exactly equal to the old per-delta concatenation - blocks, order, fields, usage, stop fields - and raise the
same exceptions at the same delta.  The reference implementation is inlined from f50eb8f6/6fd236c9."""
import copy
import json
import unittest

from serve.server import anthropic_collect


def reference_collect(events):
    """The pre-change collector, verbatim (6fd236c9)."""
    msg, blocks = None, []
    for item in events:
        if item is None:
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except ValueError:
                blocks.pop()
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    return msg


def ev(kind, **body):
    return (kind, body)


def ms():
    return ev("message_start", message={"id": "m1", "type": "message", "role": "assistant", "model": "s",
                                        "content": [], "stop_reason": None, "stop_sequence": None,
                                        "usage": {"input_tokens": 5, "output_tokens": 0}})


def md(reason="end_turn", out=1, seq=None):
    return ev("message_delta", delta={"stop_reason": reason, "stop_sequence": seq}, usage={"output_tokens": out})


class AnthropicCollectParity(unittest.TestCase):
    def both(self, events):
        return reference_collect(copy.deepcopy(events)), anthropic_collect(copy.deepcopy(events))

    def assertEqualBoth(self, events):
        a, b = self.both(events)
        self.assertEqual(a, b)
        return a

    def test_mixed_blocks_unicode_heartbeat_empty_fragment(self):
        events = [ms(),
                  ev("content_block_start", content_block={"type": "thinking", "thinking": ""}),
                  ev("content_block_delta", delta={"type": "thinking_delta", "thinking": "\u00e9\u4f60\U0001F469"}),
                  None,
                  ev("content_block_stop"),
                  ev("content_block_start", content_block={"type": "text", "text": "seed "}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": "a"}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": ""}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": "\u4f60"}),
                  ev("content_block_stop"),
                  ev("content_block_start", content_block={"type": "tool_use", "id": "t1", "name": "f", "input": {}}),
                  ev("content_block_delta", delta={"type": "input_json_delta", "partial_json": '{"p": "\\'}),
                  ev("content_block_delta", delta={"type": "input_json_delta", "partial_json": 'u0041", "n\\n": 1}'}),
                  ev("content_block_stop"),
                  md("tool_use", 3, seq=None)]
        out = self.assertEqualBoth(events)
        self.assertEqual([b["type"] for b in out["content"]], ["thinking", "text", "tool_use"])
        self.assertEqual(out["content"][2]["input"], {"p": "A", "n\n": 1})     # split escape parsed exactly, not repaired
        self.assertEqual(out["usage"]["output_tokens"], 3)

    def test_many_small_deltas_coalesce_identically(self):
        for n in (1, 63, 64, 65, 5000):                            # around GROUP=64 and far past it
            events = [ms(), ev("content_block_start", content_block={"type": "text", "text": ""})]
            events += [ev("content_block_delta", delta={"type": "text_delta", "text": "abcdefgh"}) for _ in range(n)]
            events += [ev("content_block_stop"), md(out=n)]
            out = self.assertEqualBoth(events)
            self.assertEqual(out["content"][0]["text"], "abcdefgh" * n)

    def test_tool_block_without_stop_keeps_raw_json_like_before(self):
        events = [ms(),
                  ev("content_block_start", content_block={"type": "tool_use", "id": "t", "name": "f", "input": {}}),
                  ev("content_block_delta", delta={"type": "input_json_delta", "partial_json": '{"a": '}),
                  md()]
        out = self.assertEqualBoth(events)
        self.assertIn("_json", out["content"][0])                  # the old leak stayed exactly as it was

    def test_malformed_tool_json_drops_the_block(self):
        events = [ms(),
                  ev("content_block_start", content_block={"type": "tool_use", "id": "t", "name": "f", "input": {}}),
                  ev("content_block_delta", delta={"type": "input_json_delta", "partial_json": '{"a": '}),
                  ev("content_block_stop"), md()]
        out = self.assertEqualBoth(events)
        self.assertEqual(out["content"], [])                       # #211: no input to give

    def test_empty_and_nonempty_initial_content(self):
        for initial in ("", "pre"):
            events = [ms(),
                      ev("content_block_start", content_block={"type": "thinking", "thinking": initial}),
                      ev("content_block_delta", delta={"type": "thinking_delta", "thinking": "post"}),
                      ev("content_block_stop"), md()]
            out = self.assertEqualBoth(events)
            self.assertEqual(out["content"][0]["thinking"], initial + "post")

    def test_delta_on_a_block_without_the_field_raises_like_before(self):
        for d in ({"type": "text_delta", "text": "x"}, {"type": "thinking_delta", "thinking": "x"}):
            events = [ms(), ev("content_block_start", content_block={"type": "tool_use", "id": "t", "name": "n"}),
                      ev("content_block_delta", delta=d)]
            for fn in (reference_collect, anthropic_collect):
                with self.subTest(delta=d["type"], fn=fn.__name__):
                    with self.assertRaises(KeyError):
                        fn(copy.deepcopy(events))

    def test_none_initial_content_raises_type_error_like_before(self):
        # the copy-probe evaluates `init += frag` at the first delta, exactly like the old `b["text"] +=`:
        # same TypeError, same stream position, without ever touching the block's own value.
        events = [ms(), ev("content_block_start", content_block={"type": "text", "text": None}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": "x"})]
        for fn in (reference_collect, anthropic_collect):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(TypeError):
                    fn(copy.deepcopy(events))

    def test_upstream_stream_exceptions_propagate_unchanged(self):
        def boom():
            yield ms()
            yield ev("content_block_start", content_block={"type": "text", "text": ""})
            raise RuntimeError("engine died")
        for fn in (reference_collect, anthropic_collect):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(RuntimeError):
                    fn(boom())

    def test_stream_without_message_start_raises_like_before(self):
        events = [ev("content_block_start", content_block={"type": "text", "text": "t"}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": "x"}), ev("content_block_stop")]
        for fn in (reference_collect, anthropic_collect):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises((AttributeError, TypeError)) as a:
                    fn(copy.deepcopy(events))
                with self.assertRaises((AttributeError, TypeError)) as b:
                    fn(copy.deepcopy(events))
                self.assertEqual(type(a.exception), type(b.exception))

    def test_duplicate_stop_and_mixed_delta_types_per_block(self):
        events = [ms(), ev("content_block_start", content_block={"type": "text", "text": "", "thinking": ""}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": "a"}),
                  ev("content_block_delta", delta={"type": "thinking_delta", "thinking": "b"}),
                  ev("content_block_delta", delta={"type": "input_json_delta", "partial_json": '{"k": 1}'}),
                  ev("content_block_stop"), ev("content_block_stop"), md("end_turn", 3)]
        out = self.assertEqualBoth(events)
        self.assertEqual(out["content"], [{"type": "text", "text": "a", "thinking": "b", "input": {"k": 1}}])

    def test_caller_event_streams_mutated_identically(self):
        a = [ms(), ev("content_block_start", content_block={"type": "text", "text": "t"}),
             ev("content_block_delta", delta={"type": "text_delta", "text": "x"}),
             ev("content_block_stop"), md()]
        b = copy.deepcopy(a)
        ra, rb = reference_collect(a), anthropic_collect(b)
        self.assertEqual(ra, rb)
        self.assertEqual(json.dumps(a, ensure_ascii=True, sort_keys=True, default=str),
                         json.dumps(b, ensure_ascii=True, sort_keys=True, default=str))   # same mutation, no extra

    def test_anomalous_types_fail_at_their_delta_not_later(self):
        # Codex review 2026-10-21 (collector-review-614810cf): with a raising event SOURCE the reference
        # dies with the field's TypeError at the bad delta; a deferred flush would instead surface the
        # source's RuntimeError and consume further events. Must be the TypeError, for both.
        def source(fn_field, bad_at):
            events = [ms(), ev("content_block_start", content_block={"type": "thinking", "thinking": ""})]
            for i in range(bad_at):
                events.append(ev("content_block_delta", delta={"type": "thinking_delta", "thinking": "ok"}))
            events.append(ev("content_block_delta", delta={"type": "thinking_delta", "thinking": 7}))
            it = iter(events)
            for i, e in enumerate(it):
                yield e
            raise RuntimeError("upstream error")     # only reached if the bad delta was swallowed

        for fn in (reference_collect, anthropic_collect):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(TypeError):
                    fn(source("thinking", 2))
        # first partial_json=None: the old `b.get("_json","") + None` failed at THAT delta
        events = [ms(), ev("content_block_start", content_block={"type": "tool_use", "id": "t", "name": "f", "input": {}}),
                  ev("content_block_delta", delta={"type": "input_json_delta", "partial_json": None}),
                  md()]
        for fn in (reference_collect, anthropic_collect):
            with self.subTest(json_none=fn.__name__):
                with self.assertRaises(TypeError):
                    fn(copy.deepcopy(events))

    def test_odd_first_fragment_on_tolerant_initial_matches_old(self):
        # Claude review claude-pointer-and-collect-review-20261021: a non-string FIRST fragment on an
        # initial whose += tolerates it (list, int) used to enter the buffer - only _json checked this.
        # A: list initial + bytes first delta SUCCEEDS in the old code (bytes extend to ints); a joined
        # buffer instead raised TypeError at the flush.
        events = [ms(), ev("content_block_start", content_block={"type": "text", "text": ["L"]}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": b"by"}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": "a"}),
                  ev("content_block_stop"), md(out=2)]
        out = self.assertEqualBoth(events)
        self.assertEqual(out["content"][0]["text"], ["L", 98, 121, "a"])
        # B: int initial + odd chain: exception TYPE and MESSAGE and the number of events CONSUMED from
        # the source must all match - the deferred flush shifted the failure one delta late and pulled
        # one extra event out of the source.
        b2 = [ms(),
              ev("content_block_start", content_block={"type": "thinking", "thinking": 5}),
              ev("content_block_delta", delta={"type": "thinking_delta", "thinking": 5}),
              ev("content_block_delta", delta={"type": "thinking_delta", "thinking": "a"}),
              ev("content_block_delta", delta={"type": "thinking_delta"}),   # missing key, must be reached
              md(out=3)]

        def outcome(fn, events):
            used = [0]

            def source():
                for e in events:
                    used[0] += 1
                    yield e
            try:
                fn(source())
                return ("ok", used[0])
            except Exception as exc:
                return (type(exc).__name__, str(exc), used[0])

        self.assertEqual(outcome(reference_collect, copy.deepcopy(b2)),
                         outcome(anthropic_collect, copy.deepcopy(b2)))

    def test_forged_json_initial_is_never_mutated_in_place(self):
        # the old JSON line REBUILT with +; a forged list initial value shared with the caller's event must
        # stay untouched (the earlier generic += flush extended it). The run itself ends in json.loads
        # raising for both arms - what matters is WHERE and that the caller's list survived.
        block = {"type": "tool_use", "id": "t", "name": "f", "input": {}, "_json": []}
        events = [ms(), ev("content_block_start", content_block=block),
                  ev("content_block_delta", delta={"type": "input_json_delta", "partial_json": ["x"]}),
                  ev("content_block_stop"), md("tool_use")]
        for fn in (reference_collect, anthropic_collect):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(TypeError):
                    fn(copy.deepcopy(events))
        with self.assertRaises(TypeError):                   # same end point for the real-events run
            anthropic_collect(events)                        # (json.loads at the stop, as the reference)
        self.assertEqual(block["_json"], [])                 # caller object untouched by the new code
        for fn in (reference_collect, anthropic_collect):    # dict initial + str frag: + fails AT the delta
            with self.subTest(dict_init=fn.__name__):
                with self.assertRaises(TypeError):
                    fn([ms(), ev("content_block_start", content_block={"type": "tool_use", "id": "t", "name": "f",
                                                                       "input": {}, "_json": {}}),
                        ev("content_block_delta", delta={"type": "input_json_delta", "partial_json": "x"}),
                        ev("content_block_stop"), md()])

    def test_buffered_strings_then_list_fragment_keep_delta_order(self):
        events = [ms(), ev("content_block_start", content_block={"type": "text", "text": []}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": "ab"}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": ["c"]}),
                  ev("content_block_stop"), md()]
        out = self.assertEqualBoth(events)
        self.assertEqual(out["content"][0]["text"], ["a", "b", "c"])

    def test_forged_nonstring_content_matches_iadd_semantics(self):
        # the production emitter only ever sends str; forged events must still behave like the old
        # collector's __iadd__: list initial + list fragments SUCCEEDED there and still succeed here.
        events = [ms(), ev("content_block_start", content_block={"type": "text", "text": ["L"]}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": ["a", "b"]}),
                  ev("content_block_delta", delta={"type": "text_delta", "text": ["c"]}),
                  ev("content_block_stop"), md()]
        self.assertEqualBoth(events)
        events = [ms(), ev("content_block_start", content_block={"type": "text", "text": []})]
        events += [ev("content_block_delta", delta={"type": "text_delta", "text": [str(i)]}) for i in range(65)]
        events += [ev("content_block_stop"), md(out=65)]                 # generic chain across the group boundary
        out = self.assertEqualBoth(events)
        self.assertEqual(out["content"][0]["text"], [str(i) for i in range(65)])
        # list initial + str fragment: list.__iadd__ ITERATES the string ([] += "x" -> ['x']) - the old
        # collector did exactly that, and the generic replay must reproduce it rather than raising.
        out = self.assertEqualBoth([ms(), ev("content_block_start", content_block={"type": "text", "text": []}),
                                    ev("content_block_delta", delta={"type": "text_delta", "text": "xy"}),
                                    ev("content_block_stop"), md()])
        self.assertEqual(out["content"][0]["text"], ["x", "y"])

    def test_no_shared_buffers_between_requests(self):
        e1 = [ms(), ev("content_block_start", content_block={"type": "text", "text": ""}),
              ev("content_block_delta", delta={"type": "text_delta", "text": "one"}), ev("content_block_stop"), md()]
        e2 = [ms(), ev("content_block_start", content_block={"type": "text", "text": ""}),
              ev("content_block_delta", delta={"type": "text_delta", "text": "two"}), ev("content_block_stop"), md()]
        a, b = anthropic_collect(e1), anthropic_collect(e2)
        self.assertEqual((a["content"][0]["text"], b["content"][0]["text"]), ("one", "two"))


if __name__ == "__main__":
    unittest.main()
