"""A03f - the BT token line is only a token if it is exactly what the NATIVE emitter can print.

Contract (fixed BEFORE the tests; decisions marked): the emitter prints
`std::printf("BT %d %d\\n", b, (int) y)` (generate.cpp:10997 CPU and :11172 batch; SYCL identical),
and the sampler's argmax yields y in [0, n_vocab) (sampler.dp.cpp:176-180, 262-265) - so a valid BT
line has EXACTLY three fields: "BT", the slot as canonical non-negative decimal (no leading zeros,
no sign), and the token the same way, never above INT32_MAX (%d's own range - decision: an INT32
ceiling IS emitter-grammar (printf %d cannot print more); NO vocabulary-size cap, the server must not
assume the pack's vocab).  Anything else is stream corruption and is NEVER a token: it may not enter
out/btw (fabricated token), never surfaces as a client 400 (ValueError/IndexError from a parse land
in the 400 handlers today - the client would be blamed for an engine-side wire fault), never leaks
busy/claim, never touches a follower.  The request ends as a server-side stream-integrity error
(503 family), and its finally keeps running the UNTOUCHED A03b chain (arm, BSTOP once, watcher);
EOF / a valid BDONE / the quarantine decide - no new lifecycle policy.
Watcher side: its guarded BT-tail parse resets tail=[] on a corrupt token line - conservative, no
positive cache; pinned as-is (A03b r06 contract), NOT "repaired".  Decision on retention: a corrupt
line CONSUMED by the decode arm (before any watcher attaches) leaves a later watcher's engine-confirmed
retention intact - the fault was on the wire, the engine's state is healthy, A03b retention parity.
Helper: BT lines are skipped (never end, never crash) - pinned.

Fixtures: FAKE_BATCH opt-in `--bt-text/--bt-arabic/--bt-negative/--bt-extra/--bt-trunc` (one corrupt
TOKEN word at produced==3, engine alive) and `--bt-eofcut` (complete record WITHOUT newline, then the
process dies - a whole trailing line IS a real token) / `--bt-eoftrunc` (token field missing at
death - genuinely truncated).  Defaults byte-identical.
Baseline control: against the unfixed tree the five corrupt-word rows go red BY BEHAVIOUR (400 blamed
on the client, or a full 200 answer with a FABRICATED token - int('٧') == 7 parses, int('-5') parses);
the two EOF rows and the helper/watcher rows are pins green on both sides.
"""
import queue
import threading
import time
import unittest
import urllib.error
from unittest import mock

import serve.test_parallel as tp
from serve import server


class BtFieldMatrix(unittest.TestCase):
    """the grammar gate itself - the emitter prints exactly three fields, %d canonical decimals."""

    def field(self, line, slot=1):
        f = line.split()
        gate = getattr(server.StrataEngine, "_bt_field", None)
        self.assertIsNotNone(gate, "A03f: a BT grammar gate must exist (shape pin)")
        return gate(f, slot)

    def test_valid_words(self):
        for line, want in (("BT 1 257", 257), ("BT 1 0", 0), ("BT 1 2147483647", 2147483647),
                           ("BT  1 7", 7)):       # doubled space collapses in the caller's split():
                                                  #  accepted decision (printf never prints one), pinned
            self.assertEqual(self.field(line), want, f"native-printable: {line!r}")

    def test_corrupt_words_are_never_tokens(self):
        for line in ("BT 1 x", "BT 1 \u00b2", "BT 1 \u0667",            # int('\u0667')==7 PARSES - the poison
                     "BT 1 -5", "BT 1 007", "BT 1 1_0", "BT 1 2.0",     # sign / leading zero / underscore
                     "BT 1 2147483648", "BT 1 " + "9" * 6000,           # beyond %d / CPython digit limit
                     "BT 1 7 9", "BT 1", "BT 1 ", "BT 0 7", "BT -1 7",
                     "BT 1 +7", "BT 01 7"):
            self.assertIsNone(self.field(line), f"not emitter-grammar: {line!r}")

    def test_slot_binding_rechecked_at_the_consumer(self):
        self.assertIsNone(self.field("BT 0 7", slot=1), "the pump routed it, the consumer still binds")


class HelperSkipsBt(unittest.TestCase):
    """the A03d read-back helper: BT lines (valid or corrupt) never end and never crash it."""

    def fixture(self):
        e = object.__new__(server.StrataEngine)
        e.gen, e.slot_q = 0, [queue.Queue() for _ in range(2)]
        e.slot_cv = threading.Condition()
        return e

    def test_corrupt_bt_line_is_skipped_then_the_bdone_answers(self):
        e = self.fixture()
        e.slot_q[1].put("BT 1 x")                              # corrupt token word: not an end, no crash
        e.slot_q[1].put("BT 1 7")                              # valid token line: also not the helper's business
        e.slot_q[1].put("BDONE 1 3 stop 1.0")
        got = e._bdone_after_eos(1, 0.5)
        self.assertEqual(got[0], "bdone")


class WatcherTailReset(unittest.TestCase):
    """watcher tail collection: a corrupt token line resets the tail - conservative, pinned as-is."""

    def arm(self, stream=None, batch=2):
        e = self.e = object.__new__(server.StrataEngine)
        e.gen, e.slot_q = 0, [queue.Queue() for _ in range(batch)]
        e.slot_busy = [False] * batch
        e.slot_busy[0] = True
        e.slot_held = [[] for _ in range(batch)]
        e.slot_releasing, e.slot_quarantine = set(), set()
        e.slot_cv = threading.Condition()
        e.sent = []
        e._send = lambda line: e.sent.append(line)
        e._release_slot_when_done(0, stream, born=0)
        return e

    def test_corrupt_bt_resets_the_tail_so_no_positive_cache_is_minted(self):
        e = self.arm(stream=[10, 11])
        e.slot_q[0].put("BT 0 12")
        e.slot_q[0].put("BT 0 x")                              # corrupt: the tail is no longer trustworthy
        time.sleep(0.6)                                        # the watcher poll is 5 s: 0.6 s proves it
        with e.slot_cv:                                        #  keeps WATCHING (the A03b r06 inner guard -
            self.assertTrue(e.slot_busy[0],                    #  without that guard only the outer r04 error
                             "a corrupt token line is not an end: the watcher keeps watching")  # net would
            self.assertEqual(e.slot_quarantine, set(),         #  mark quarantine: the inner guard does not)
                             "the inner guard keeps the plain watch - no quarantine, no replacement watcher")
        e.slot_q[0].put("BDONE 0 3 stop 5.0")                  # a VALID end follows
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with e.slot_cv:
                if not e.slot_busy[0]:
                    break
            time.sleep(0.02)
        with e.slot_cv:
            self.assertFalse(e.slot_busy[0], "the valid record still ends the wait")
            self.assertEqual(e.slot_held[0], [],
                             "tail reset: no positive cache may cover a stream that carried corruption")


class HarnessBt(unittest.TestCase):
    """full real legs over HTTP: a corrupt token word is an ENGINE-side fault - never a client 400,
    never a fabricated token, and the A03b state chain (arm/waiter/EOF) stays intact."""

    def setUp(self):
        self.r = None

    def start(self, flag):
        self.r = tp.ParallelService()
        self.r.start(2, slot_cache=True, reuse=True, more=("--readfield", flag))
        self.e = self.r.engine

    def tearDown(self):
        if self.r:
            self.r.tearDown()

    def two_chats(self):
        res = {}
        with mock.patch.object(server, "EOS_IDS", {257}):
            ta = threading.Thread(target=lambda: res.setdefault("a", self._chat("List three primes please")))
            tb = threading.Thread(target=lambda: res.setdefault("b", self._chat("Name two rivers in Germany")))
            ta.start(); tb.start(); ta.join(40); tb.join(40)
        return res

    def _chat(self, text):
        try:
            return self.r.chat(text, 200)
        except urllib.error.HTTPError as ex:
            return ("http", ex.code, ex.read().decode(errors="replace"))
        except Exception as ex:                                # noqa: BLE001
            return ("gone", type(ex).__name__)

    def assert_engine_side_503(self, res):
        for k in ("a", "b"):
            self.assertIn(k, res, "both requests answered (no hang)")
            v = res[k]
            self.assertIsInstance(v, tuple, f"leg {k}: corruption must end the request with an error")
            self.assertEqual(v[1], 503,
                             f"leg {k}: an engine wire fault must be 5xx server_error, got {v[1]}: {v[2][:120]}")
            self.assertIn("server_error", v[2], "not invalid_request_error - never the client's fault")
            self.assertNotIn("invalid literal", v[2], "no parse internals leaked to the client")

    def wait_closed(self, secs=8):
        e = self.e
        end = time.monotonic() + secs
        while time.monotonic() < end:
            with e.slot_cv:
                if not any(e.slot_busy):
                    return True
            time.sleep(0.05)
        return False

    def assert_state_clean(self):
        # claims/releasing/busy must be clean.  slot_held is NOT asserted empty here on purpose
        # (decision, A03b parity): the corrupt line was CONSUMED by the decode arm before any watcher
        # attached, so the watcher that later closes the slot caches the engine-CONFIRMED tail of a
        # stream whose engine state was never touched (the fault was on the wire, not in the engine) -
        # exactly A03b's healthy retention.  A corrupt line that passes the WATCHER itself resets its
        # tail to [] - pinned behaviourally in WatcherTailReset.
        e = self.e
        with e.slot_cv:
            self.assertEqual(sorted(e.slot_claims), [], "no claim leaks")
            self.assertEqual(sorted(e.slot_releasing), [], "no watcher left dangling")

    # ---- red on the parent BY BEHAVIOUR --------------------------------------

    def test_nonnumeric_token_word_is_an_engine_fault_not_a_client_400(self):
        # parent: int('x') ValueError escapes the decode -> the 400 handler blames the CLIENT.
        self.start("--bt-text")
        self.assert_engine_side_503(self.two_chats())
        self.assertTrue(self.wait_closed(), "the untouched A03b chain closes the slot (real BDONE follows)")
        self.assert_state_clean()

    def test_unicode_digit_token_word_never_fabricates_a_token(self):
        # parent: int('\u0667') == 7 PARSES -> a full 200 answer containing a FABRICATED token (the real
        # token at that position was different) - a silently poisoned response.
        self.start("--bt-arabic")
        self.assert_engine_side_503(self.two_chats())
        self.assertTrue(self.wait_closed(), "A03b chain closes the slot after the abort")
        self.assert_state_clean()

    def test_negative_token_word_never_fabricates_a_token(self):
        # parent: int('-5') parses - the sampler argmax yields [0, n_vocab), printf prints this only for
        # a negative id the sampler cannot produce: grammar violation, and the tokenizer then decodes -5.
        self.start("--bt-negative")
        self.assert_engine_side_503(self.two_chats())
        self.assertTrue(self.wait_closed())
        self.assert_state_clean()

    def test_extra_field_is_not_the_emitter_grammar(self):
        # printf("BT %d %d\\n") never prints a fourth field. Parent: silently ACCEPTED (token parsed,
        # full 200 answer) - every other consumer of this line family is strict on field count.
        self.start("--bt-extra")
        self.assert_engine_side_503(self.two_chats())
        self.assertTrue(self.wait_closed())
        self.assert_state_clean()

    def test_missing_token_field_midstream_is_not_an_indexerror_400(self):
        # parent: line.split()[2] IndexError -> the 400 handler prints "the request is malformed
        # (IndexError...)" - the client is blamed for the engine's broken pipe.
        self.start("--bt-trunc")
        self.assert_engine_side_503(self.two_chats())
        self.assertTrue(self.wait_closed())
        self.assert_state_clean()

    def test_truncated_token_field_at_death_is_not_a_client_400(self):
        # the process dies right after "BT <b> " - the pump yields the unterminated remainder: NO token
        # field at all. Parent: IndexError -> 400 blamed on the client; the engine death also EOFs the
        # other leg (a legit 503 there). After the gate: engine-side 503 for this leg too.
        self.start("--bt-eoftrunc")
        self.assert_engine_side_503(self.two_chats())
        self.assertTrue(self.wait_closed(), "EOF (the engine died) closes both slots conservatively")
        self.assert_state_clean()

    # ---- pins: green before AND after (behaviour must NOT change) -------------

    def test_complete_line_without_newline_at_eof_is_a_real_token(self):
        # the pump yields the whole trailing line "BT 1 <tok>" - complete record, valid token; then EOF
        # (the engine died) ends the decode as an engine fault. Pin: classification and closure stay.
        self.start("--bt-eofcut")
        res = self.two_chats()
        for k in ("a", "b"):
            self.assertIn(k, res)
            v = res[k]
            self.assertIsInstance(v, tuple, "the engine died: an error response")
            self.assertEqual(v[1], 503, f"leg {k}: a dead engine is server_error, got {v[1]}")
            self.assertIn("engine stopped", v[2], "the honest engine-death message (EOF path)")
        self.assertTrue(self.wait_closed(), "EOF closed both slots (watcher / pump)")
        self.assert_state_clean()


if __name__ == "__main__":
    unittest.main()
