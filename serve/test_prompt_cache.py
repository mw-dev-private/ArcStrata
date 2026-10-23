"""The prompt / token-id cache (tools/strata_tokenizer.PromptEncoder behind Service.encode_prompt) against a fresh UNCACHED tokenisation (A08a): exact id lists after every step of
randomised conversations (special literals, partial specials that straddle turns, private-use characters, plain spans, store on/off, several conversations), kept entries intact, the caller owns
every returned list, threads, and the invalidation contract.  CPU only; the byte tokenizer, and the pack's tokenizer when one is installed (skipped otherwise).

Tests marked POLICY pin current behaviour that is an open question (the cache size is set once at construction; forgetting a refused turn also empties the conversation; the budget counts
entries, not tokens); they exist so a change is noticed, not because it is endorsed."""
import json
import os
import random
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve import server  # noqa: E402
from serve.frontend import LITERAL_TAGS, ChatTemplate  # noqa: E402


class BT(server.ByteTokenizer):
    @property
    def max_special_len(self):
        return max(len(s) for s in self.SPECIALS)


class PrefixSpecials(BT):
    """a vocabulary where one special is the start of another (longest listed first, as a regex alternation sorted by length): the case the resume margin exists for.
    The shipped pack has no such pair (checked: 33 specials, none a prefix of another, none overlapping), so only this tokenizer can show the margin at work."""
    SPECIALS = ["<|ab|>c", "<|ab|>"] + server.ByteTokenizer.SPECIALS


def pack_tokenizer():
    for c in [os.environ.get("STRATA_TOKENIZER", "")] + [str(p) for p in ROOT.glob("packs/*/tokenizer")] + [str(p) for p in Path("/home/markus/strata/data/packs").glob("*/tokenizer")]:
        if c and (Path(c) / "vocab.json").exists():
            t = Path(c)
            vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
            toks = [None] * len(vocab)
            for s, i in vocab.items():
                toks[i] = s
            return ST.Tokenizer(toks, (t / "merges.txt").read_text(encoding="utf-8").split("\n"), json.loads((t / "token_type.json").read_text())), t
    return None, None


PIECES = ["<|ab|>", "<|ab|>c", "c", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<|im_", "<|im_s", "tart|>", "end|>", "<|", "|>", "hello", " world", "\n", "def f(x):\n    return x\n",
          "你好", "\U0001f30a", "", "", "     ", "1234567890", "user\n", "assistant\n"]


def oracle(tok, text, plain=()):
    return tok.encode(text, parse_special=True, plain=plain) if plain else tok.encode(text, parse_special=True)


def entries_intact(case, tok, enc):
    for text, plain, ids, ends, counts in enc.entries:
        want, marks = tok.encode_marked(text, parse_special=True, plain=plain)
        case.assertEqual(ids, want)
        case.assertEqual((ends, counts), ([m[0] for m in marks], [m[1] for m in marks]))


class Exactness(unittest.TestCase):
    def fuzz(self, tok, seeds, steps):
        for keep in (1, 2, 4):
            for seed in seeds:
                rng = random.Random(seed * 10 + keep)
                enc = ST.PromptEncoder(tok, keep=keep)
                prev = [None, None]
                for step in range(steps):
                    c = rng.randint(0, 1)
                    p = prev[c]
                    op = rng.choice(["extend", "extend", "same", "truncate", "edit", "new", "straddle"])
                    if p is None or op == "new":
                        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 10)))
                    elif op == "extend":
                        text = p + "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 5)))
                    elif op == "same":
                        text = p
                    elif op == "truncate":
                        text = p[:rng.randint(0, len(p))]
                    elif op == "edit":
                        a = rng.randint(0, len(p))
                        text = p[:a] + rng.choice(PIECES) + p[min(len(p), a + rng.randint(0, 8)):]
                    else:
                        text = p + rng.choice(["<|im_", "<|im_s", "<|"])
                    plain = ()
                    if rng.random() < .35 and text:
                        a = rng.randrange(len(text))
                        plain = ((a, min(len(text), a + rng.randint(1, 25))),)
                    got = enc.encode(text, plain, rng.random() > .25)
                    self.assertEqual(got, oracle(tok, text, plain), (keep, seed, step, op))
                    entries_intact(self, tok, enc)
                    self.assertLessEqual(len(enc.entries), keep)
                    prev[c] = text

    def test_byte_tokenizer_random_conversations(self):
        self.fuzz(BT(), range(6), 120)

    def test_pack_tokenizer_random_conversations(self):
        tok, where = pack_tokenizer()
        if tok is None:
            self.skipTest("no pack tokenizer here")
        self.fuzz(tok, range(2), 60)

    def test_a_longer_special_that_starts_where_a_kept_one_ends(self):
        """kept text ends in '<|ab|>' (the short special); the next turn continues with 'c', which makes it the LONG special '<|ab|>c': resuming at the end of the short one would be wrong"""
        tok = PrefixSpecials()
        enc = ST.PromptEncoder(tok, keep=4)
        base = "<|im_start|>user\n" + "word " * 30 + "<|im_end|>\n<|im_start|>assistant\nsee <|ab|>"
        enc.encode(base, (), True)
        for tail in ("c rest", "d rest", "", "c"):
            text = base + tail
            self.assertEqual(enc.encode(text, (), True), oracle(tok, text), tail)
        self.fuzz(tok, range(3), 80)

    def test_a_special_that_straddles_two_turns(self):
        tok = BT()
        enc = ST.PromptEncoder(tok, keep=4)
        base = "<|im_start|>user\n" + "word " * 30 + "<|im_start|>user\nlast <|im_"
        enc.encode(base, (), True)
        for tail in ("end|>\n", "start|>x", "e", ""):
            text = base + tail
            self.assertEqual(enc.encode(text, (), True), oracle(tok, text), tail)

    def test_service_encode_prompt_equals_the_same_service_without_the_cache(self):
        tok = BT()
        server.Service.start_telemetry = lambda self: None

        class E:
            max_context = known_ctx = 10 ** 6
        tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        cached, plain_svc = server.Service(E(), tok, tpl), server.Service(E(), tok, tpl)
        plain_svc.prompts = None
        bits = list(LITERAL_TAGS.values()) + list(LITERAL_TAGS.keys()) + ["hello", "\n", "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|im_", "end|>", "你", ""]
        rng = random.Random(4)
        msgs = [{"role": "user", "content": "start"}]
        for step in range(150):
            op = rng.choice(["grow", "grow", "same", "edit", "shorter"])
            if op == "grow":
                msgs.append({"role": "assistant" if msgs[-1]["role"] == "user" else "user", "content": "".join(rng.choice(bits) for _ in range(rng.randint(1, 5)))})
            elif op == "edit":
                i = rng.randrange(len(msgs))
                msgs[i] = {"role": msgs[i]["role"], "content": "".join(rng.choice(bits) for _ in range(rng.randint(1, 5)))}
            elif op == "shorter" and len(msgs) > 1:
                del msgs[rng.randint(1, len(msgs) - 1):]
            kw = {"enable_thinking": rng.random() > .5}
            store = rng.random() > .25
            self.assertEqual(cached.encode_prompt([dict(m) for m in msgs], None, dict(kw), store), plain_svc.encode_prompt([dict(m) for m in msgs], None, dict(kw), store), (step, op))
            entries_intact(self, tok, cached.prompts)


class Ownership(unittest.TestCase):
    TEXT = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\n" + "word " * 100 + "<|im_end|>\n<|im_start|>assistant\n"      # as the template renders it: it ends with the generation prompt

    def test_the_caller_owns_every_returned_list(self):
        tok = BT()
        enc = ST.PromptEncoder(tok, keep=4)
        a = enc.encode(self.TEXT, (), True)
        before = [(e[0], list(e[2]), list(e[3]), list(e[4])) for e in enc.entries]
        a.append(1)
        a[0] = -1
        a.reverse()
        self.assertEqual([(e[0], e[2], e[3], e[4]) for e in enc.entries], before)
        b = enc.encode(self.TEXT, (), True)
        self.assertEqual(b, oracle(tok, self.TEXT))
        self.assertIsNot(b, a)
        c = enc.encode(self.TEXT + "x", (), False)
        c.clear()
        self.assertTrue(all(x is not e[2] for x in (a, b, c) for e in enc.entries))
        self.assertEqual(len(enc.entries), 1)                       # store=False wrote nothing

    def test_threads_with_a_barrier_get_exact_results(self):
        tok = BT()
        for threads, keep in ((4, 2), (8, 4)):
            enc = ST.PromptEncoder(tok, keep=keep)
            bar, errors = threading.Barrier(threads), []

            def work(t):
                try:
                    for r in range(60):
                        bar.wait(10)
                        text = self.TEXT + f"<|im_start|>assistant\nanswer {t} {r}"
                        got = enc.encode(text, (), r % 3 != 0)
                        if got != oracle(tok, text):
                            errors.append((t, r))
                        if r % 4 == 1:
                            enc.forget(got)
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))
            ts = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
            [t.start() for t in ts]
            [t.join(60) for t in ts]
            self.assertEqual(errors, [])
            self.assertLessEqual(len(enc.entries), keep)
            entries_intact(self, tok, enc)

    def test_forget_removes_the_entry_with_equal_ids(self):
        tok = BT()
        enc = ST.PromptEncoder(tok, keep=4)
        r = enc.encode(self.TEXT, (), True)
        enc.forget(list(r))
        self.assertEqual(enc.entries, [])


class Invalidation(unittest.TestCase):
    def service(self, ctx, starting=False):
        server.Service.start_telemetry = lambda self: None

        class E:
            max_context = known_ctx = ctx

            def alive(self):
                return True
        e = E()
        e.starting = starting
        return server.Service(e, BT(), ChatTemplate(ROOT / "serve/chat_template.jinja"))

    MSGS = [{"role": "user", "content": "hello " * 20}]

    def test_contract(self):
        ok = self.service(100000)
        ids, _, _ = ok.prepare(self.MSGS, None, {}, 5)
        self.assertEqual(len(ok.prompts.entries), 1)                  # accepted: kept
        refused = self.service(len(ids) + 8 + 2)
        with self.assertRaises(ValueError):
            refused.prepare(self.MSGS, None, {}, 50)
        self.assertEqual(refused.prompts.entries, [])                  # a context refusal forgets it
        starting = self.service(100000)
        starting.engine.max_context = starting.engine.known_ctx = 0
        starting.engine.starting = True
        with self.assertRaises(server.EngineStarting):
            starting.prepare(self.MSGS, None, {}, 5)
        self.assertEqual(len(starting.prompts.entries), 1)             # an engine that is starting keeps it
        counted = self.service(100000)
        counted.encode_prompt(self.MSGS, None, {}, store=False)
        self.assertEqual(counted.prompts.entries, [])                  # a token count writes nothing

    def test_POLICY_the_size_is_fixed_at_construction(self):
        class Late:
            max_context = known_ctx = 1000
            spawn = ("x", [])
            info = {}
        server.Service.start_telemetry = lambda self: None
        svc = server.Service(Late(), BT(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.engine.info = {"conversation_cache_slots": 16}
        self.assertEqual(svc.prompts.keep, 4)                          # slots the engine reports later do not resize the cache

        class Spawn:
            max_context = known_ctx = 1000
            spawn = ("x", ["--conversation-cache-slots", "7"])
            info = {}
        self.assertEqual(server.Service(Spawn(), BT(), ChatTemplate(ROOT / "serve/chat_template.jinja")).prompts.keep, 7)

    def test_POLICY_forgetting_a_refused_turn_leaves_the_conversation_with_no_kept_prompt(self):
        tok = BT()
        enc = ST.PromptEncoder(tok, keep=4)
        t1 = "".join(f"<|im_start|>user\nquestion {i} " + "w " * 10 + "<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>\n" for i in range(30))
        t2 = t1 + "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n"
        enc.encode(t1, (), True)
        i2 = enc.encode(t2, (), True)
        self.assertEqual([e[0] for e in enc.entries], [t2])            # the continuing prompt replaced its predecessor
        enc.forget(i2)
        self.assertEqual(enc.entries, [])                              # the refused turn took the conversation's entry with it (ids stay exact; only a re-encode is paid)

    def test_POLICY_an_identical_prompt_replaces_itself_only_through_the_reuse_rule(self):
        """'continues' means: more than half of the kept text is reusable up to a special boundary at least max_special_len-1 before the end. A rendered prompt ends with the generation prompt, so a
        resend replaces its predecessor; a raw text ending in <|im_end|>\\n (the template never renders that) does not, and identical copies fill the slots."""
        tok = BT()
        enc = ST.PromptEncoder(tok, keep=4)
        for _ in range(4):
            enc.encode(Ownership.TEXT, (), True)
        self.assertEqual(len(enc.entries), 1)
        enc = ST.PromptEncoder(tok, keep=4)
        raw = Ownership.TEXT[:-len("<|im_start|>assistant\n")]
        for _ in range(4):
            enc.encode(raw, (), True)
        self.assertEqual((len(enc.entries), len({e[0] for e in enc.entries})), (4, 1))

    def test_POLICY_the_budget_counts_entries_not_tokens(self):
        tok = BT()
        enc = ST.PromptEncoder(tok, keep=3)
        for k in range(5):
            enc.encode(f"<|im_start|>system\nconversation {k}<|im_end|>\n" + "x" * (1000 * (k + 1)), (), True)
        self.assertEqual(len(enc.entries), 3)                          # bounded by keep only, whatever the prompt sizes


if __name__ == "__main__":
    unittest.main()
