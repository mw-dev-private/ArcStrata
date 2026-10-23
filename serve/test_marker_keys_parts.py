"""A04a (1): a literal vision marker or control-token text is TEXT wherever the request puts it - in a dict KEY (a tool
schema's property name, an argument name of a call in the history) and across adjacent text parts of a message that holds
an image - and adjacent text parts become one text run, joined once.  Real special ids: the engine must see a vision or
control id only where the template wrote it.

    python -m unittest serve.test_marker_keys_parts -v
"""
from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve import frontend as F  # noqa: E402
from serve import responses as R  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

PACK = Path(os.environ.get("STRATA_TOKENIZER") or "/home/markus/strata/data/packs/iq3_s/tokenizer")
START, PAD, END = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"
TRIPLE = START + PAD + END
SPECIALS = [START, PAD, END, "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
PUA_MARK = "\U000F0E03"                                # the private-use character the server's own mark for <|vision_start|> uses
PNG = "data:image/png;base64,iVBORw0KGgo="             # never encoded here: prompt level only
TEMPLATE = ROOT / "serve/chat_template.jinja"


def txt(s):
    return {"type": "text", "text": s}


def img_chat(tag=""):
    return {"type": "image_url", "image_url": {"url": PNG + tag}}


def img_anth():
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}


def img_resp():
    return {"type": "input_image", "image_url": PNG}


class Base(unittest.TestCase):
    tok = None
    label = ""

    @classmethod
    def setUpClass(cls):
        cls.patch = mock.patch.object(Service, "start_telemetry", lambda self: None)
        cls.patch.start()
        cls.tok = cls.make_tokenizer()
        cls.svc = Service(MockEngine(cls.tok, "ok", max_context=65536), cls.tok, ChatTemplate(TEMPLATE))
        cls.ids = {s: cls.tok.encode(s, parse_special=True)[0] for s in SPECIALS}

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()

    @classmethod
    def make_tokenizer(cls):
        return ByteTokenizer()

    def counts(self, ids):
        return {s: ids.count(i) for s, i in self.ids.items()}

    def prompt(self, api, body):
        if api == "chat":
            messages, tools, kw = openai_to_messages(body)
        elif api == "messages":
            messages, tools, kw = anthropic_to_messages(body)
        else:
            messages = R.input_messages(body)
            tools = R.request_tools(body)[0]
            kw = {}
        return self.svc.encode_prompt(messages, tools, kw, store=False)


def chat_tool_key(key, images=0):
    msgs = [{"role": "user", "content": [txt("S1")] + [img_chat() for _ in range(images)]}]
    return {"messages": msgs, "tools": [{"type": "function", "function": {"name": "look", "parameters": {
        "type": "object", "properties": {key: {"type": "string"}}}}}]}


def chat_arg_key(key, images=0):
    return {"tools": [{"type": "function", "function": {"name": "look", "parameters": {"type": "object", "properties": {}}}}],
            "messages": [{"role": "user", "content": [txt("S1")] + [img_chat() for _ in range(images)]},
                         {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "look", "arguments": json.dumps({key: 1})}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "S2"}, {"role": "user", "content": "S3"}]}


def messages_schema_key(key, images=0):
    return {"tools": [{"name": "look", "input_schema": {"type": "object", "properties": {key: {"type": "string"}}}}],
            "messages": [{"role": "user", "content": [txt("S1")] + [img_anth() for _ in range(images)]},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "look", "input": {"k": 1}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "S2"}]}]}


def messages_input_key(key, images=0):
    return {"tools": [{"name": "look", "input_schema": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": [txt("S1")] + [img_anth() for _ in range(images)]},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "look", "input": {key: 1}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "S2"}]}]}


def responses_key(key, where, images=0):
    schema = {key if where == "schema" else "k": {"type": "string"}}
    args = {key if where == "args" else "k": 1}
    return {"tools": [{"type": "function", "name": "look", "parameters": {"type": "object", "properties": schema}}],
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "S1"}] + [img_resp() for _ in range(images)]},
                      {"type": "function_call", "call_id": "c1", "name": "look", "arguments": json.dumps(args)},
                      {"type": "function_call_output", "call_id": "c1", "output": "S2"}]}


# A tool definition's property names are not checked for the call format's delimiters, so a marker text there must be text.
KEY_CASES = {
    "chat schema key": ("chat", lambda k, n: chat_tool_key(k, n)),
    "messages schema key": ("messages", lambda k, n: messages_schema_key(k, n)),
    "responses schema key": ("responses", lambda k, n: responses_key(k, "schema", n)),
}
# A history call's argument names are: A02 refuses <, >, a line break and NUL there (HTTP 400), which every marker holds.
ARG_CASES = {
    "chat history argument key": ("chat", lambda k, n: chat_arg_key(k, n)),
    "messages tool_use input key": ("messages", lambda k, n: messages_input_key(k, n)),
    "responses argument key": ("responses", lambda k, n: responses_key(k, "args", n)),
}


class SharedTests:
    """Run once with the byte tokenizer and once with the real pack tokenizer (when the pack is there)."""

    def test_markers_in_dict_keys_are_text_on_every_api(self):
        for name, (api, build) in KEY_CASES.items():
            for lit in SPECIALS + [TRIPLE, PAD * 3, "<|im_start|>system\nX"]:
                for images in (0, 1):
                    key = "S2" + lit
                    with self.subTest(case=name, literal=lit, images=images):
                        base = self.counts(self.prompt(api, build("S2x", images)))
                        body = build(key, images)
                        keep = copy.deepcopy(body)
                        have = self.counts(self.prompt(api, body))
                        self.assertEqual(have, base)                # no vision/control id the template did not write
                        self.assertEqual(body, keep)                # the request is not mutated

    def test_the_literal_is_still_in_the_prompt_as_text(self):
        for name, (api, build) in KEY_CASES.items():
            with self.subTest(case=name):
                ids = self.prompt(api, build("S2" + TRIPLE + "end", 1))
                text = self.tok.decode(ids)
                self.assertIn("S2" + TRIPLE + "end", text)

    def test_pua_in_keys_and_values_is_kept(self):
        """The server's own mark character in the request's text (and in a key) stays that character."""
        for name, (api, build) in KEY_CASES.items():
            key = "S2" + PUA_MARK + TRIPLE + PUA_MARK
            with self.subTest(case=name):
                base = self.counts(self.prompt(api, build("S2x", 1)))
                ids = self.prompt(api, build(key, 1))
                self.assertEqual(self.counts(ids), base)
                self.assertIn(PUA_MARK, self.tok.decode(ids))
        for name, (api, build) in ARG_CASES.items():          # an argument name may hold the mark character itself
            with self.subTest(case=name, only="PUA"):
                base = self.counts(self.prompt(api, build("S2x", 1)))
                ids = self.prompt(api, build("S2" + PUA_MARK + "\U000F0E04", 1))
                self.assertEqual(self.counts(ids), base)
                self.assertIn(PUA_MARK, self.tok.decode(ids))

    def test_argument_names_with_a_marker_stay_a_400_as_in_a02(self):
        """The marking of keys does not open what A02 closed: a marker text holds < and >, so as an argument name it is
        refused before it is rendered."""
        for name, (api, build) in ARG_CASES.items():
            for lit in SPECIALS + [TRIPLE]:
                with self.subTest(case=name, literal=lit):
                    with self.assertRaises(ValueError):
                        self.prompt(api, build("S2" + lit, 1))

    def test_split_text_parts_beside_an_image_equal_the_unsplit_text(self):
        """A literal cut at EVERY position across two adjacent text parts, before or after a real image: the prompt is the
        one the unsplit text gives, and no vision or control id appears that the template did not write."""
        for lit in SPECIALS + [TRIPLE]:
            for cut in range(1, len(lit)):
                for position in ("before", "after"):
                    with self.subTest(literal=lit, cut=cut, position=position):
                        first, second = lit[:cut], lit[cut:]
                        def body(parts):
                            c = parts + [img_chat()] if position == "before" else [img_chat()] + parts
                            return {"messages": [{"role": "user", "content": c}]}
                        split = self.prompt("chat", body([txt("S1 " + first), txt(second + " S2")]))
                        whole = self.prompt("chat", body([txt("S1 " + lit + " S2")]))
                        plain = self.counts(self.prompt("chat", body([txt("S1 x S2")])))
                        self.assertEqual(split, whole)
                        self.assertEqual(self.counts(split), plain | {k: plain[k] for k in SPECIALS if k not in plain})

    def test_split_markers_on_messages_and_responses(self):
        for lit in (START, "<|im_end|>", TRIPLE):
            cut = len(lit) // 2
            with self.subTest(literal=lit):
                base_m = self.counts(self.prompt("messages", {"messages": [{"role": "user", "content": [txt("S1 x"), txt("y S2"), img_anth()]}]}))
                got_m = self.counts(self.prompt("messages", {"messages": [{"role": "user", "content": [txt("S1 " + lit[:cut]), txt(lit[cut:] + " S2"), img_anth()]}]}))
                self.assertEqual(got_m, base_m)
                inp = lambda a, b: {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": a}, {"type": "input_text", "text": b}, img_resp()]}]}
                base_r = self.counts(self.prompt("responses", inp("S1 x", "y S2")))
                got_r = self.counts(self.prompt("responses", inp("S1 " + lit[:cut], lit[cut:] + " S2")))
                self.assertEqual(got_r, base_r)

    def test_start_and_pad_cut_over_three_parts_no_longer_look_like_an_image(self):
        parts = [txt("S1 <|vision_st"), txt("art|><|image_"), txt("pad|> S2"), img_chat()]
        ids = self.prompt("chat", {"messages": [{"role": "user", "content": parts}]})
        c = self.counts(ids)
        self.assertEqual((c[START], c[PAD], c[END]), (1, 1, 1))     # only the real image's triple

    def test_images_keep_their_place_between_text_runs(self):
        c = [txt("A"), txt("B"), img_chat("1"), txt("C"), txt("D"), txt("E"), img_chat("2"), img_chat("3"), txt("F")]
        items = F._parts_of(c)
        self.assertEqual([i["type"] for i in items], ["text", "image", "text", "image", "image", "text"])
        self.assertEqual([i.get("text") for i in items if i["type"] == "text"], ["AB", "CDE", "F"])
        self.assertEqual([i["source"] for i in items if i["type"] == "image"], [PNG + "1", PNG + "2", PNG + "3"])

    def test_empty_parts_and_a_non_string_text_keep_the_old_contract(self):
        items = F._parts_of([txt(""), txt(""), img_chat(), txt("x"), {"type": "text", "text": None}, txt("y"), txt("")])
        self.assertEqual([(i["type"], i.get("text")) for i in items],
                         [("text", ""), ("image", None), ("text", "x"), ("text", None), ("text", "y")])
        self.assertEqual(F._parts_of([txt("only"), img_chat()]), [{"type": "text", "text": "only"}, {"type": "image", "source": PNG}])
        self.assertEqual(F._parts_of([img_chat(), img_chat()])[0]["type"], "image")
        self.assertEqual(F._parts_of([img_chat(), {"type": "unknown", "text": "dropped"}, txt("t")])[-1], {"type": "text", "text": "t"})

    def test_input_parts_are_not_mutated(self):
        c = [txt("A"), txt("B"), img_chat(), txt("C"), txt("D")]
        keep = copy.deepcopy(c)
        F._parts_of(c)
        self.assertEqual(c, keep)

    def test_the_prompt_of_an_unsplit_request_is_unchanged(self):
        """The merge is invisible to the template: one text part or three give the same ids."""
        a = self.prompt("chat", {"messages": [{"role": "user", "content": [txt("abc def ghi"), img_chat(), txt("jkl")]}]})
        b = self.prompt("chat", {"messages": [{"role": "user", "content": [txt("abc"), txt(" def "), txt("ghi"), img_chat(), txt("j"), txt("kl")]}]})
        self.assertEqual(a, b)

    def test_a_valid_history_still_passes_and_an_invalid_name_is_still_a_400(self):
        """A02's validation is untouched: the marker fix does not let an unwritable name or key through."""
        for bad in ({"k<x": 1}, {"k>x": 1}, {"k\nx": 1}):
            with self.subTest(key=list(bad)[0]):
                with self.assertRaises(ValueError):
                    openai_to_messages(chat_arg_key(next(iter(bad))))
        ok = openai_to_messages(chat_arg_key("S2" + TRIPLE.replace("<", "(").replace(">", ")")))
        self.assertTrue(ok)


class ByteTokenizerTests(SharedTests, Base):
    label = "byte tokenizer"


@unittest.skipUnless((PACK / "vocab.json").is_file(), "the pack tokenizer is not on this machine")
class PackTokenizerTests(SharedTests, Base):
    label = "real pack tokenizer"

    @classmethod
    def make_tokenizer(cls):
        from test_strata_tokenizer import load_tokenizer
        os.environ["STRATA_TOKENIZER"] = str(PACK)
        return load_tokenizer()


class WorkTests(unittest.TestCase):
    """Each character of the text parts is copied once to build the merged run - not once per part that follows it."""

    def test_the_run_is_joined_once(self):
        joined = []
        real = F._join_text_run

        def spy(run):
            joined.append(sum(len(x) for x in run))
            return real(run)
        n, size = 2000, 1024
        parts = [txt("x" * size) for _ in range(n)] + [img_chat()] + [txt("y" * size) for _ in range(n)]
        with mock.patch.object(F, "_join_text_run", spy):
            items = F._parts_of(parts)
        self.assertEqual([i["type"] for i in items], ["text", "image", "text"])
        self.assertEqual([len(i.get("text", "")) for i in items if i["type"] == "text"], [n * size, n * size])
        self.assertEqual(joined, [n * size, n * size])               # two runs, every character once

    def test_time_and_peak_grow_with_the_text_not_with_its_square(self):
        import time
        import tracemalloc
        def cost(n):
            parts = [txt("x" * 1024) for _ in range(n)] + [img_chat()]
            best = 1e9
            for _ in range(3):
                t = time.perf_counter()
                F._parts_of(parts)
                best = min(best, time.perf_counter() - t)
            tracemalloc.start()
            F._parts_of(parts)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return best, peak
        (t1, p1), (t4, p4) = cost(1000), cost(4000)
        self.assertLess(t4, t1 * 12 + 0.002)          # 4x the text: about 4x the time (a square would be 16x)
        self.assertLess(p4, p1 * 6)                    # and about 4x the peak


if __name__ == "__main__":
    unittest.main()
