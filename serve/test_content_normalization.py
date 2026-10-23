"""Roles, content parts and tool history through the three request converters (A07c): the common subset gives ONE normal form on chat / messages / responses and equals an independent
expectation; the documented differences and the findings of the audit are pinned per route.  CPU only; no engine.

Tests marked POLICY pin current behaviour that is an open question (route differences or silent content loss); they exist so a change is noticed, not because it is endorsed."""
import copy
import json
import threading
import unittest
from pathlib import Path

from serve import frontend as F
from serve import responses as R
from serve import server

ROOT = Path(__file__).resolve().parents[1]


def parts_chat(c):
    if isinstance(c, str): return c
    out = []
    for kind, v in c:
        out.append({"type": "text", "text": v} if kind == "text" else {"type": "image_url", "image_url": {"url": v}})
    return out
def parts_msgs(c):
    if isinstance(c, str): return c
    return [{"type": "text", "text": v} if k == "text" else {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": v.split(",", 1)[1]}} for k, v in c]
def parts_resp(c, role):
    if isinstance(c, str): return c
    return [{"type": "output_text" if role == "assistant" else "input_text", "text": v} if k == "text" else {"type": "input_image", "image_url": v} for k, v in c]
def raw_args(a): return a if isinstance(a, str) else json.dumps(a)

def to_chat(spec):
    msgs = []
    for it in spec:
        r = it[0]
        if r in ("system", "developer", "user"): msgs.append({"role": r, "content": parts_chat(it[1])})
        elif r == "assistant":
            m = {"role": "assistant", "content": parts_chat(it[1]) if it[1] is not None else None}
            if len(it) > 2 and it[2]: m["tool_calls"] = [{"id": cid, "type": "function", "function": {"name": n, "arguments": raw_args(a)}} for cid, n, a in it[2]]
            if len(it) > 3 and it[3]: m["reasoning_content"] = it[3]
            msgs.append(m)
        elif r == "tool": msgs.append({"role": "tool", "tool_call_id": it[1], "content": parts_chat(it[2])})
    return {"model": "x", "messages": msgs}
def to_messages(spec):
    system, msgs, pending = [], [], []
    for it in spec:
        r = it[0]
        if r in ("system", "developer"): system.append(it[1]); continue
        if r != "tool" and r != "user" and pending: msgs.append({"role": "user", "content": pending}); pending = []      # results sit in the user message right after the assistant turn
        if r == "user":
            txt = [{"type": "text", "text": it[1]}] if isinstance(it[1], str) else parts_msgs(it[1])
            if pending or not isinstance(it[1], str): msgs.append({"role": "user", "content": pending + txt}); pending = []
            else: msgs.append({"role": "user", "content": it[1]})
        elif r == "assistant":
            blocks = []
            if len(it) > 3 and it[3]: blocks.append({"type": "thinking", "thinking": it[3], "signature": ""})
            if it[1]: blocks += [{"type": "text", "text": it[1]}] if isinstance(it[1], str) else parts_msgs(it[1])
            for cid, n, a in (it[2] if len(it) > 2 and it[2] else []): blocks.append({"type": "tool_use", "id": cid, "name": n, "input": a if isinstance(a, dict) else a})
            msgs.append({"role": "assistant", "content": blocks if blocks else ""})
        elif r == "tool": pending.append({"type": "tool_result", "tool_use_id": it[1], "content": it[2] if isinstance(it[2], str) else parts_msgs(it[2])})
    if pending: msgs.append({"role": "user", "content": pending})
    out = {"model": "x", "max_tokens": 50, "messages": msgs}
    if system: out["system"] = "\n\n".join(system) if len(system) > 1 else system[0]
    return out
def to_responses(spec):
    items = []
    for it in spec:
        r = it[0]
        if r in ("system", "developer", "user"): items.append({"type": "message", "role": r, "content": parts_resp(it[1], r)})
        elif r == "assistant":
            if len(it) > 3 and it[3]: items.append({"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": it[3]}]})
            if it[1] is not None: items.append({"type": "message", "role": "assistant", "content": parts_resp(it[1], "assistant")})
            for cid, n, a in (it[2] if len(it) > 2 and it[2] else []): items.append({"type": "function_call", "call_id": cid, "name": n, "arguments": raw_args(a)})
        elif r == "tool": items.append({"type": "function_call_output", "call_id": it[1], "output": it[2] if isinstance(it[2], str) else parts_resp(it[2], "user")})
    return {"model": "x", "input": items}
BUILD = {"chat": to_chat, "messages": to_messages, "responses": to_responses}

# ---- the independent expectation for the SUPPORTED common subset (text, images as given, tool calls with dict args)
def expected(spec, route):
    out = []; first = True
    for it in spec:
        r = it[0]
        role = "system" if r == "developer" else r
        if role == "system":
            if route == "messages": continue                               # Anthropic's system is a field, handled below
            out.append({"role": "system" if not out else "user", "content": it[1]}); continue
        if role == "user":
            out.append({"role": "user", "content": norm_content(it[1])}); continue
        if role == "assistant":
            m = {"role": "assistant", "content": norm_content(it[1]) if it[1] is not None else ""}
            if len(it) > 3 and it[3]: m["reasoning_content"] = it[3]
            calls = it[2] if len(it) > 2 and it[2] else []
            if calls: m["tool_calls"] = [{"function": {"name": n, "arguments": a if isinstance(a, dict) else {}}} for cid, n, a in calls]
            out.append(m); continue
        if role == "tool": out.append({"role": "tool", "content": norm_content(it[2])})
    if route == "messages":
        systems = [it[1] for it in spec if it[0] in ("system", "developer")]
        if systems: out.insert(0, {"role": "system", "content": "\n\n".join(systems) if len(systems) > 1 else systems[0]})
    return out
def norm_content(c):
    if isinstance(c, str): return c
    if not any(k == "image" for k, v in c): return "".join(v for k, v in c)
    items, run = [], []
    for k, v in c:
        if k == "text": run.append(v)
        else:
            if run: items.append({"type": "text", "text": "".join(run)}); run = []
            items.append({"type": "image", "source": v})
    if run: items.append({"type": "text", "text": "".join(run)})
    return items


def convert(route, payload):
    before = copy.deepcopy(payload)
    if route == "chat":
        out = F.openai_to_messages(payload)[0]
    elif route == "messages":
        out = F.anthropic_to_messages(payload)[0]
    else:
        out = R.input_messages(payload)
    assert payload == before, "the converter changed its input"
    return out


def U(c):
    return {"role": "user", "content": c}


def A(c, calls=None, **k):
    return {"role": "assistant", "content": c, **({"tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]} if calls else {}), **k}


def T(c):
    return {"role": "tool", "content": c}


def S(c):
    return {"role": "system", "content": c}


IMG = "data:image/png;base64,QUJD"
SPECS = {
    "user text": [("user", "hello")], "system + user": [("system", "be brief"), ("user", "hi")], "developer + user": [("developer", "rules"), ("user", "hi")],
    "multi-turn": [("system", "s"), ("user", "a"), ("assistant", "b"), ("user", "c")],
    "call, result, answer": [("user", "weather?"), ("assistant", "let me check", [("c1", "get_weather", {"city": "Zurich"})]), ("tool", "c1", "sunny"), ("assistant", "It is sunny."), ("user", "thanks")],
    "parallel calls in order": [("user", "go"), ("assistant", "", [("a", "f", {"x": 1}), ("b", "g", {"y": 2})]), ("tool", "a", "RA"), ("tool", "b", "RB"), ("assistant", "done")],
    "text part list": [("user", [("text", "one "), ("text", "two")])],
    "adjacent texts around an image": [("user", [("text", "a"), ("text", "b"), ("image", IMG), ("text", "c"), ("text", "d")])], "text image text": [("user", [("text", "look "), ("image", IMG), ("text", " here")])],
    "tool result with an image": [("user", "x"), ("assistant", "", [("c1", "shot", {})]), ("tool", "c1", [("text", "result "), ("image", IMG)])],
    "reasoning history": [("user", "q"), ("assistant", "answer", None, "thought it through"), ("user", "next")],
    "assistant without content": [("user", "q"), ("assistant", None, [("c1", "f", {"a": "b"})]), ("tool", "c1", "R")],
    "unicode, markers, think literal": [("user", "caf\u00e9 \u4e2d\u6587 \U0001f30a \ue000 <|im_start|> <think>x</think> <|im_")],
    "empty assistant content": [("user", "hi"), ("assistant", ""), ("user", "again")],
}


class CommonSubset(unittest.TestCase):
    def test_every_route_gives_the_independent_expectation(self):
        for name, spec in SPECS.items():
            for route in ("chat", "messages", "responses"):
                with self.subTest(case=name, route=route):
                    self.assertEqual(convert(route, BUILD[route](spec)), expected(spec, route))

    def test_a_late_system_message_becomes_a_user_message_in_place(self):
        spec = [("user", "a"), ("assistant", "b"), ("system", "hook"), ("user", "c")]
        for route in ("chat", "responses"):
            self.assertEqual(convert(route, BUILD[route](spec)), expected(spec, route))
        got = convert("messages", {"model": "x", "max_tokens": 5, "messages": [U("a"), A("b"), S("hook"), U("c")]})
        self.assertEqual([m["role"] for m in got], ["user", "assistant", "user", "user"])


class TheRouteDiffers(unittest.TestCase):
    """documented differences between the routes (the code says so)"""

    def test_leading_system_messages(self):
        self.assertEqual(convert("chat", {"messages": [S("A"), S("B"), U("q")]}), [S("A"), U("B"), U("q")])
        self.assertEqual(convert("responses", {"input": [{"type": "message", "role": "system", "content": "A"}, {"type": "message", "role": "developer", "content": "B"}, {"type": "message", "role": "user", "content": "q"}]}), [S("A\n\nB"), U("q")])
        self.assertEqual(convert("responses", {"instructions": "A", "input": [{"type": "message", "role": "developer", "content": "B"}, {"type": "message", "role": "user", "content": "q"}]}), [S("A\n\nB"), U("q")])
        self.assertEqual(convert("messages", {"max_tokens": 5, "system": [{"type": "text", "text": "A "}, {"type": "text", "text": "B"}], "messages": [U("q")]}), [S("A B"), U("q")])

    def test_unsupported_parts_are_left_out_on_chat_and_messages_and_refused_on_responses(self):
        self.assertEqual(convert("chat", {"messages": [U([{"type": "text", "text": "see"}, {"type": "file", "file": {"file_id": "f"}}])]}), [U("see")])
        self.assertEqual(convert("messages", {"max_tokens": 5, "messages": [U([{"type": "text", "text": "see"}, {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AA"}}])]}), [U("see")])
        with self.assertRaises(R.ResponsesError) as c:
            R.input_messages({"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "see"}, {"type": "input_file", "file_id": "f"}]}]})
        self.assertEqual(c.exception.param, "input[0].content[1].type")
        self.assertEqual(convert("chat", {"messages": [U([{"type": "file", "file": {"file_id": "f"}}])]}), [U("")])

    def test_refusal_parts_are_read_as_their_text(self):
        self.assertEqual(convert("chat", {"messages": [U("q"), {"role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}, U("again")]}), [U("q"), A("no"), U("again")])
        self.assertEqual(convert("responses", {"input": [{"type": "message", "role": "user", "content": "q"}, {"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}, {"type": "message", "role": "user", "content": "again"}]}), [U("q"), A("no"), U("again")])

    def test_tool_results_in_the_order_of_the_calls_on_every_route(self):
        """results sent as b, a reach the model as a, b on chat, messages and responses when they exactly answer the calls (F2; the unknown-id and partial cases: test_tool_result_order)"""
        calls = [{"id": "a", "function": {"name": "f", "arguments": "{}"}}, {"id": "b", "function": {"name": "g", "arguments": "{}"}}]
        chat = convert("chat", {"messages": [U("go"), {"role": "assistant", "content": None, "tool_calls": calls}, {"role": "tool", "tool_call_id": "b", "content": "RB"}, {"role": "tool", "tool_call_id": "a", "content": "RA"}]})
        self.assertEqual([m["content"] for m in chat if m["role"] == "tool"], ["RA", "RB"])
        mess = convert("messages", {"max_tokens": 5, "messages": [U("go"), {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}, {"type": "tool_use", "id": "b", "name": "g", "input": {}}]},
                                                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "RB"}, {"type": "tool_result", "tool_use_id": "a", "content": "RA"}]}]})
        self.assertEqual([m["content"] for m in mess if m["role"] == "tool"], ["RA", "RB"])
        resp = convert("responses", {"input": [{"type": "message", "role": "user", "content": "go"}, {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"}, {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"},
                                              {"type": "function_call_output", "call_id": "b", "output": "RB"}, {"type": "function_call_output", "call_id": "zz", "output": "RZ"}, {"type": "function_call_output", "call_id": "a", "output": "RA"}]})
        self.assertEqual([m["content"] for m in resp if m["role"] == "tool"], ["RA", "RB", "RZ"])         # responses: unknown ids last, the sent order otherwise

    def test_a_user_text_beside_tool_results_comes_after_them_on_messages(self):
        base = [U("q"), {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "f", "input": {}}]}]
        got = convert("messages", {"max_tokens": 5, "messages": base + [{"role": "user", "content": [{"type": "text", "text": "note"}, {"type": "tool_result", "tool_use_id": "c", "content": "r"}]}]})
        self.assertEqual(got[2:], [T("r"), U("note")])

    def test_history_arguments_are_kept_as_text_when_they_are_not_an_object(self):
        def chat(raw):
            return convert("chat", {"messages": [U("q"), {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": raw}}]}, {"role": "tool", "content": "r"}]})[1]["tool_calls"][0]["function"]["arguments"]

        def resp(raw):
            return convert("responses", {"input": [{"type": "message", "role": "user", "content": "q"}, {"type": "function_call", "call_id": "c", "name": "f", "arguments": raw}, {"type": "function_call_output", "call_id": "c", "output": "r"}]})[1]["tool_calls"][0]["function"]["arguments"]
        for raw, want in (("{\"a\": 1}", {"a": 1}), ("{\"a\": ", {"arguments": "{\"a\": "}), ("", {}), ("  ", {})):
            self.assertEqual(chat(raw), want, raw)
            self.assertEqual(resp(raw), want, raw)
        self.assertEqual(chat("[1, 2]"), {"arguments": "[1, 2]"})
        self.assertEqual(resp("[1, 2]"), {"arguments": [1, 2]})                  # POLICY: responses keeps the parsed value, chat / messages the text

    def test_call_names_with_delimiters_are_refused_everywhere(self):
        with self.assertRaises(ValueError):
            F.openai_to_messages({"messages": [U("q"), {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "function": {"name": "a<b", "arguments": "{}"}}]}]})
        with self.assertRaises(R.ResponsesError):
            R.input_messages({"input": [{"type": "message", "role": "user", "content": "q"}, {"type": "function_call", "call_id": "c", "name": "a\nb", "arguments": "{}"}]})


class SilentContentLossPOLICY(unittest.TestCase):
    """shapes whose content disappears without an error on some route (audit findings; the same shapes are refused or kept on another route)"""

    def test_messages_system_as_a_list_of_plain_strings_becomes_empty(self):
        self.assertEqual(convert("messages", {"max_tokens": 5, "system": ["RULE-A", "RULE-B"], "messages": [U("q")]}), [S(""), U("q")])

    def test_chat_user_content_as_one_part_object_or_a_list_of_strings_becomes_empty(self):
        self.assertEqual(convert("chat", {"messages": [U({"type": "text", "text": "MARKER-X"})]}), [U("")])
        self.assertEqual(convert("chat", {"messages": [U(["MARKER-X", "MARKER-Y"])]}), [U("")])
        self.assertEqual(convert("responses", {"input": [{"type": "message", "role": "user", "content": ["MARKER-X", "MARKER-Y"]}]}), [U("MARKER-XMARKER-Y")])
        with self.assertRaises(R.ResponsesError):
            R.input_messages({"input": [{"type": "message", "role": "user", "content": {"type": "input_text", "text": "MARKER-X"}}]})

    def test_images_by_role(self):
        sysimg_chat = convert("chat", {"messages": [{"role": "system", "content": [{"type": "text", "text": "SYS"}, {"type": "image_url", "image_url": {"url": IMG}}]}, U("q")]})
        self.assertEqual(sysimg_chat[0], S("SYS"))                                                      # dropped on chat (and on messages)
        sysimg_resp = convert("responses", {"input": [{"type": "message", "role": "system", "content": [{"type": "input_text", "text": "SYS"}, {"type": "input_image", "image_url": IMG}]}, {"type": "message", "role": "user", "content": "q"}]})
        self.assertIsInstance(sysimg_resp[0]["content"], list)                                           # kept by the converter on responses ...
        class E:
            max_context = known_ctx = 10 ** 6
        server.Service.start_telemetry = lambda self: None
        svc = server.Service(E(), server.ByteTokenizer(), server.ChatTemplate(ROOT / "serve/chat_template.jinja"))
        with self.assertRaises(Exception) as c:                                                         # ... and refused when the template renders it
            svc.render_prompt(sysimg_resp, None, {})
        self.assertIn("System message cannot contain images", str(c.exception))
        asst_chat = convert("chat", {"messages": [U("q"), {"role": "assistant", "content": [{"type": "text", "text": "ASSIST"}, {"type": "image_url", "image_url": {"url": IMG}}]}, U("more")]})
        self.assertIsInstance(asst_chat[1]["content"], list)                                             # kept on chat / messages
        asst_resp = convert("responses", {"input": [{"type": "message", "role": "user", "content": "q"}, {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ASSIST"}, {"type": "input_image", "image_url": IMG}]}, {"type": "message", "role": "user", "content": "more"}]})
        self.assertEqual(asst_resp[1], A("ASSIST"))                                                      # dropped on responses

    def test_an_empty_image_source_is_taken_on_chat_and_refused_on_responses(self):
        self.assertEqual(convert("chat", {"messages": [U([{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": ""}}])]})[0]["content"][1], {"type": "image", "source": ""})
        with self.assertRaises(R.ResponsesError) as c:
            R.input_messages({"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}, {"type": "input_image", "image_url": ""}]}]})
        self.assertEqual(c.exception.param, "input[0].content[1]")


class IsolationPOLICY(unittest.TestCase):
    def test_no_converter_changes_a_plain_request(self):
        for name, spec in SPECS.items():
            for route in ("chat", "messages", "responses"):
                convert(route, BUILD[route](spec))                                                       # convert() asserts the request is unchanged

    def test_anthropic_writes_the_thinking_budget_into_the_request(self):
        p = {"model": "x", "max_tokens": 9, "thinking": {"type": "enabled", "budget_tokens": 77}, "messages": [U("q")]}
        F.anthropic_to_messages(p)
        self.assertEqual(p["reasoning_budget_tokens"], 77)                                               # the one known writer

    def test_the_stable_tool_order_is_process_global(self):
        def tools(names):
            return F.openai_to_messages({"messages": [U("q")], "tools": [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}} for n in names]})[1]
        F._TOOL_ORDERS.clear()
        tools(["zz_a", "zz_b", "zz_c"])
        self.assertEqual([t["name"] for t in tools(["zz_c", "zz_a", "zz_b"])], ["zz_a", "zz_b", "zz_c"])
        self.assertEqual([t["name"] for t in tools(["zz_a", "zz_a", "zz_b"])], ["zz_a", "zz_a", "zz_b"])

    def test_threads_sharing_parts_get_one_result(self):
        part = {"type": "text", "text": "shared <think>q</think>"}
        payload = {"messages": [{"role": "user", "content": [part, part]}, {"role": "assistant", "content": [part]}, {"role": "user", "content": [part, part]}]}
        want = F.openai_to_messages(copy.deepcopy(payload))[0]
        errors, bar = [], threading.Barrier(6)

        def work():
            try:
                for _ in range(60):
                    bar.wait(10)
                    if F.openai_to_messages(payload)[0] != want:
                        errors.append("mismatch")
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
        ts = [threading.Thread(target=work) for _ in range(6)]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
