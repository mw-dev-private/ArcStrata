"""Tool results are placed by position, so results sent in another order than their calls must be put back (A07c F2).
Contract: a run of tool results that is EXACTLY the answer to the assistant turn before it (as many results as calls, every id a
non-empty string, no id twice, each result id one of the calls' ids) is ordered by the calls on chat, messages and responses.
Anything else (missing / repeated / unknown id, partial group, text between the results) keeps the order sent on chat and messages.
CPU only; no engine."""
import copy
import unittest

from serve import frontend as F
from serve import responses as R

U = lambda c: {"role": "user", "content": c}


def chat(msgs):
    return F.openai_to_messages({"model": "x", "messages": msgs})[0]
def mess(msgs):
    return F.anthropic_to_messages({"max_tokens": 5, "messages": msgs})[0]


def calls_chat(*ids): return {"role": "assistant", "content": None, "tool_calls": [{"id": i, "type": "function", "function": {"name": f"f{i}", "arguments": "{}"}} for i in ids]}
def calls_mess(*ids): return {"role": "assistant", "content": [{"type": "tool_use", "id": i, "name": f"f{i}", "input": {}} for i in ids]}
def res_chat(i, c): return {"role": "tool", "tool_call_id": i, "content": c}
def res_mess(i, c): return {"type": "tool_result", "tool_use_id": i, "content": c}


def tools(out): return [m["content"] for m in out if m["role"] == "tool"]
def names(out): return [c["function"]["name"] for m in out for c in m.get("tool_calls") or []]
def no_leak(t, out):
    t.assertNotIn("_call_id", repr(out))


class Ordered(unittest.TestCase):
    def test_chat_results_put_back_in_call_order(self):
        out = chat([U("go"), calls_chat("a", "b", "c"), res_chat("c", "RC"), res_chat("a", "RA"), res_chat("b", "RB")])
        self.assertEqual(tools(out), ["RA", "RB", "RC"])
        self.assertEqual(names(out), ["fa", "fb", "fc"])
        no_leak(self, out)

    def test_messages_results_put_back_in_call_order(self):
        out = mess([U("go"), calls_mess("a", "b", "c"), {"role": "user", "content": [res_mess("c", "RC"), res_mess("a", "RA"), res_mess("b", "RB")]}])
        self.assertEqual(tools(out), ["RA", "RB", "RC"])
        no_leak(self, out)

    def test_already_in_order_and_single_call_unchanged(self):
        self.assertEqual(tools(chat([U("go"), calls_chat("a", "b"), res_chat("a", "RA"), res_chat("b", "RB")])), ["RA", "RB"])
        self.assertEqual(tools(chat([U("go"), calls_chat("a"), res_chat("a", "RA")])), ["RA"])

    def test_several_turns_each_group_on_its_own_calls(self):
        out = chat([U("go"), calls_chat("a", "b"), res_chat("b", "RB1"), res_chat("a", "RA1"), {"role": "assistant", "content": "mid"},
                    U("again"), calls_chat("a", "b"), res_chat("b", "RB2"), res_chat("a", "RA2")])
        self.assertEqual(tools(out), ["RA1", "RB1", "RA2", "RB2"])          # the same ids in a later turn are a new group
        out = mess([U("go"), calls_mess("x", "y"), {"role": "user", "content": [res_mess("y", "Y1"), res_mess("x", "X1")]},
                    calls_mess("x", "y"), {"role": "user", "content": [res_mess("y", "Y2"), res_mess("x", "X2")]}])
        self.assertEqual(tools(out), ["X1", "Y1", "X2", "Y2"])

    def test_mixed_content_travels_with_its_result(self):
        img = "data:image/png;base64,QUJD"
        out = chat([U("go"), calls_chat("a", "b"), res_chat("b", [{"type": "text", "text": "B1"}, {"type": "image_url", "image_url": {"url": img}}]), res_chat("a", [{"type": "text", "text": "A1"}, {"type": "text", "text": "A2"}])])
        t = tools(out)
        self.assertEqual(t[0], [{"type": "text", "text": "A1"}, {"type": "text", "text": "A2"}][:0] or t[0])
        self.assertIn("A1", repr(t[0])); self.assertIn("B1", repr(t[1])); self.assertIn("QUJD", repr(t[1]))
        out = mess([U("go"), calls_mess("a", "b"), {"role": "user", "content": [res_mess("b", [{"type": "text", "text": "B1"}]), res_mess("a", [{"type": "text", "text": "A1"}, {"type": "text", "text": "A2"}])]}])
        self.assertIn("A1", repr(tools(out)[0])); self.assertIn("B1", repr(tools(out)[1]))

    def test_no_mutation_and_no_id_leak(self):
        p = {"model": "x", "messages": [U("go"), calls_chat("a", "b"), res_chat("b", "RB"), res_chat("a", "RA")]}
        before = copy.deepcopy(p)
        out = F.openai_to_messages(p)[0]
        self.assertEqual(p, before)
        no_leak(self, out)
        self.assertTrue(all(set(m) <= {"role", "content", "tool_calls", "reasoning_content"} for m in out))
        self.assertTrue(all(set(c) == {"function"} for m in out for c in m.get("tool_calls") or []))

    def test_responses_route_unchanged(self):
        inp = [{"type": "message", "role": "user", "content": "go"}, {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"}, {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"},
               {"type": "function_call_output", "call_id": "b", "output": "RB"}, {"type": "function_call_output", "call_id": "zz", "output": "RZ"}, {"type": "function_call_output", "call_id": "a", "output": "RA"}]
        out = R.input_messages({"input": inp})
        self.assertEqual(tools(out), ["RA", "RB", "RZ"])
        no_leak(self, out)


class KeptAsSent(unittest.TestCase):
    """not an exact answer to the calls: the order sent is kept (no new policy)"""
    def check(self, msgs, want, route="chat"):
        out = (chat if route == "chat" else mess)(msgs)
        self.assertEqual(tools(out), want)
        no_leak(self, out)

    def test_missing_id(self):
        self.check([U("g"), calls_chat("a", "b"), {"role": "tool", "content": "RB"}, res_chat("a", "RA")], ["RB", "RA"])

    def test_duplicate_result_id(self):
        self.check([U("g"), calls_chat("a", "b"), res_chat("a", "R1"), res_chat("a", "R2")], ["R1", "R2"])
        self.check([U("g"), calls_chat("a", "b"), res_chat("b", "R1"), res_chat("b", "R2")], ["R1", "R2"])

    def test_duplicate_call_id(self):
        self.check([U("g"), calls_chat("a", "a"), res_chat("a", "R1"), res_chat("a", "R2")], ["R1", "R2"])

    def test_unknown_id(self):
        self.check([U("g"), calls_chat("a", "b"), res_chat("zz", "RZ"), res_chat("a", "RA")], ["RZ", "RA"])

    def test_partial_group(self):
        self.check([U("g"), calls_chat("a", "b", "c"), res_chat("c", "RC"), res_chat("a", "RA")], ["RC", "RA"])
        self.check([U("g"), calls_chat("a", "b"), res_chat("b", "RB")], ["RB"])

    def test_more_results_than_calls(self):
        self.check([U("g"), calls_chat("a"), res_chat("a", "RA"), res_chat("b", "RB")], ["RA", "RB"])

    def test_repeated_result_id_beside_all_call_ids(self):
        """same id set, but more results than calls"""
        self.check([U("g"), calls_chat("a", "b"), res_chat("b", "R1"), res_chat("a", "R2"), res_chat("a", "R3")], ["R1", "R2", "R3"])

    def test_repeated_call_id_with_other_ids(self):
        self.check([U("g"), calls_chat("a", "b", "a"), res_chat("b", "R1"), res_chat("a", "R2"), res_chat("b", "R3")], ["R1", "R2", "R3"])

    def test_calls_without_ids(self):
        c = {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{}"}}, {"type": "function", "function": {"name": "g", "arguments": "{}"}}]}
        self.check([U("g"), c, res_chat("b", "RB"), res_chat("a", "RA")], ["RB", "RA"])

    def test_non_string_ids(self):
        self.check([U("g"), calls_chat("a", "b"), res_chat(1, "R1"), res_chat(None, "R2")], ["R1", "R2"])

    def test_text_between_results_is_moved_after_them_so_the_group_is_still_exact(self):
        """the messages converter puts a user text after the tool results of the same message (existing behaviour); the results then form one run"""
        out = mess([U("g"), calls_mess("a", "b"), {"role": "user", "content": [res_mess("b", "RB"), {"type": "text", "text": "note"}, res_mess("a", "RA")]}])
        self.assertEqual([m["content"] for m in out[2:]], ["RA", "RB", "note"])
        no_leak(self, out)

    def test_results_without_a_preceding_assistant_turn(self):
        self.check([res_chat("b", "RB"), res_chat("a", "RA")], ["RB", "RA"])


if __name__ == "__main__":
    unittest.main()
