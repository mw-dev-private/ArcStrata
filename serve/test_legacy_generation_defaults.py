"""Compatibility with the existing production request defaults."""
import unittest
from serve.frontend import anthropic_to_messages

class LegacyBudgets(unittest.TestCase):
    def test_legacy_effort_does_not_impose_budget(self):
        req = {"messages": [{"role":"user", "content":"hi"}], "thinking":{"type":"enabled", "budget_tokens":2048}}
        anthropic_to_messages(req, honor_budget=False)
        self.assertNotIn("reasoning_budget_tokens", req)
        anthropic_to_messages(req)
        self.assertEqual(req["reasoning_budget_tokens"], 2048)

    def test_explicit_budget_survives(self):
        req = {"messages": [{"role":"user", "content":"hi"}], "thinking":{"type":"enabled", "budget_tokens":2048}, "reasoning_budget_tokens":1024}
        anthropic_to_messages(req, honor_budget=False)
        self.assertEqual(req["reasoning_budget_tokens"], 1024)
