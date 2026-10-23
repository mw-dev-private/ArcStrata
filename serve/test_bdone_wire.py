"""A03e - a BDONE line is only an end record if it is a COMPLETE, plausible record.

Contract (fixed BEFORE the tests, decision points marked in REPORT): the native emitter prints
`BDONE <slot> <produced> <finish> <ms>\\n` with finish in {stop,length,cancel,nonfinite}
(generate.cpp:10987/11004 and 11164/11182, both variants identical).  A record is valid iff: five
fields, non-negative decimal slot and produced, finish in that vocabulary, a finite ms >= 0 - and for
the consumers the slot field equals the slot they are bound to.  A COMPLETE record without a trailing
newline at EOF stays valid (Codex pump probe, repeated here); a truncated one is not an end record at
all: it may not confirm an end, apply figures, mint a positive cache, send anything, or spoof a slot.
Garbage slot fields (negative/out-of-range/non-numeric) are DROPPED by the pump - decision: never
wrapped to a real slot (slot_q[-1]!) and never handed to the control stream.  A malformed line is
ignored by every consumer; the real end (valid BDONE, nonfinite, or the pump EOF) or the A03b
quarantine path takes over - no new lifecycle policy, no early release, no fabricated stats.

Fixtures: FAKE_BATCH `--garbage-bdone` (three corrupt words on the pipe - a synthetic producer:
printf %d never prints -1, this tests stream/crash protection) and `--malformed-bdone` (process dies
MID-RECORD: an unterminated truncated BDONE at EOF - the realistic crash case).
Baseline control: against the unfixed tree the pump routing rows, the watcher confirmation rows, the
helper misclassification and the decode-arm positive-cache row go red BY BEHAVIOUR.
"""
import io
import queue
import threading
import time
import types
import unittest
from unittest import mock

import serve.test_parallel as tp
from serve import server


class PumpRouting(unittest.TestCase):
    """real StrataEngine._pump over a scripted stdout - field-exact routing."""

    def pump(self, text, batch=2):
        e = object.__new__(server.StrataEngine)
        e.proc = types.SimpleNamespace(stdout=io.StringIO(text))
        e.lines = queue.Queue()
        e.slot_q = [queue.Queue() for _ in range(batch)]
        e._pump()
        def drain(q):
            out = []
            while True:
                try:
                    out.append(q.get_nowait())
                except queue.Empty:
                    return out
        return drain(e.slot_q[0]), drain(e.slot_q[1]), drain(e.lines)

    def test_complete_record_without_newline_is_valid_and_routed(self):
        q0, q1, ctl = self.pump("BDONE 0 7 stop 1")                    # EOF-mid-buffer, complete record
        self.assertEqual(q0, ["BDONE 0 7 stop 1", None], "complete at EOF stays valid (parity pin)")
        self.assertEqual(q1, [None])
        self.assertEqual(ctl, [None])                                  # the pump always EOFs `lines` too

    def test_negative_slot_is_dropped_never_wrapped(self):
        q0, q1, ctl = self.pump("BDONE -1 1 stop 1.0\nBT 0 5\nBDONE 0 1 stop 1.0\n")
        self.assertEqual(q0, ["BT 0 5\n", "BDONE 0 1 stop 1.0\n", None])   # routed lines keep the newline
        self.assertEqual(q1, [None], "slot_q[-1] wraparound must never deliver a foreign BDONE")
        self.assertEqual(ctl, [None])                                   # decision: dropped, not control-fed

    def test_out_of_range_and_non_numeric_slots_are_dropped(self):
        q0, q1, ctl = self.pump("BDONE 999 3 stop 1.0\nBDONE x 1 stop 1.0\nBT 1 5\nBDONE 1 1 stop 1.0\n")
        self.assertEqual(q0, [None])
        self.assertEqual(q1, ["BT 1 5\n", "BDONE 1 1 stop 1.0\n", None])
        self.assertEqual(ctl, [None], "a corrupt BDONE is not a control line either")


class WatcherWire(unittest.TestCase):
    """real _release_slot_when_done against scripted queues: malformed lines must not confirm, spoof,
    or mint caches; valid lines keep their exact A03b semantics."""

    def arm(self, slot=0, stream=None, batch=2):
        e = self.e = object.__new__(server.StrataEngine)
        e.gen, e.slot_q = 0, [queue.Queue() for _ in range(batch)]
        e.slot_busy = [False] * batch
        e.slot_busy[slot] = True                            # a consumer just handed THIS busy slot over
        e.slot_held = [[] for _ in range(batch)]
        e.slot_releasing, e.slot_quarantine = set(), set()
        e.slot_cv = threading.Condition()
        e.sent = []
        e._send = lambda line: e.sent.append(line)
        e._release_slot_when_done(slot, stream, born=0)
        return e

    def settle(self, secs=0.6):
        time.sleep(secs)                                        # watcher poll is 5 s: 0.6 s proves NO confirm

    def test_malformed_bdone_does_not_confirm_then_valid_closes_positive(self):
        e = self.arm(stream=[10, 11, 12])
        e.slot_q[0].put("BT 0 13")
        e.slot_q[0].put("BDONE 0 7 st")                        # truncated finish field: no end record
        self.settle()
        with e.slot_cv:
            self.assertIn(0, e.slot_releasing, "the watcher must still be armed")
            self.assertTrue(e.slot_busy[0], "a malformed line never frees a slot")
        e.slot_q[0].put("BDONE 0 7 stop 5.0")                  # the real end follows
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with e.slot_cv:
                if not e.slot_busy[0]:
                    break
            time.sleep(0.02)
        with e.slot_cv:
            self.assertFalse(e.slot_busy[0])
            self.assertEqual(e.slot_held[0], [10, 11, 12, 13][:-1],   # healthy watcher keeps retention (A03b parity)
                             "a VALID end after garbage still caches its true tail")

    def test_foreign_slot_field_never_confirms_its_watcher(self):
        e = self.arm(slot=0)
        e.slot_q[0].put("BDONE 1 7 stop 5.0")                  # well-formed but NOT this slot's record
        self.settle()
        with e.slot_cv:
            self.assertTrue(e.slot_busy[0], "slot-field mismatch: not an end record for THIS slot")
        e.slot_q[0].put(None)                                  # the real end: conservative close
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with e.slot_cv:
                if not e.slot_busy[0]:
                    break
            time.sleep(0.02)
        with e.slot_cv:
            self.assertEqual(e.slot_held[0], [], "EOF after foreign garbage: empty cache")

    def test_unicode_digit_bdone_does_not_confirm_nor_raise_then_eof_closes_empty(self):
        e = self.arm(stream=[7, 8])
        e.slot_q[0].put("BDONE 0 \u00b2 stop 1.0")             # isdigit-True, int-ValueError trap
        self.settle()
        with e.slot_cv:
            self.assertTrue(e.slot_busy[0], "still armed: malformed number ended nothing")
        e.slot_q[0].put(None)
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with e.slot_cv:
                if not e.slot_busy[0]:
                    break
            time.sleep(0.02)
        with e.slot_cv:
            self.assertEqual(e.slot_held[0], [], "EOF: conservative close, no exception escaped")

    def test_nonfinite_parity_and_garbage_then_eof_conservative(self):
        e = self.arm(stream=[1, 2, 3])
        e.slot_q[0].put("BDONE 0 0 nonfinite x")               # ms garbage -> invalid -> ignored
        self.settle()
        with e.slot_cv:
            self.assertTrue(e.slot_busy[0])
        e.slot_q[0].put(None)
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with e.slot_cv:
                if not e.slot_busy[0]:
                    break
            time.sleep(0.02)
        with e.slot_cv:
            self.assertEqual(e.slot_held[0], [])               # parity with A03b: EOF closes empty
        e2 = self.arm(stream=[4, 5, 6])
        e2.slot_q[0].put("BDONE 0 0 nonfinite 1.0")            # VALID nonfinite: ends, no positive cache
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with e2.slot_cv:
                if not e2.slot_busy[0]:
                    break
            time.sleep(0.02)
        with e2.slot_cv:
            self.assertEqual(e2.slot_held[0], [], "#879 parity stays")


class HelperWire(unittest.TestCase):
    """_bdone_after_eos: only a complete, own-slot record may become _END_BDONE."""

    def fixture(self):
        e = self.e = object.__new__(server.StrataEngine)
        e.gen, e.slot_q = 0, [queue.Queue() for _ in range(2)]
        e.slot_cv = threading.Condition()
        return e

    def payload(self, r):
        return r[1] if isinstance(r, tuple) else r             # pre-A03d baseline returned bare strings

    def test_malformed_then_valid_returns_the_valid_record(self):
        e = self.fixture()
        e.slot_q[1].put("BDONE 1 3 stop")                      # missing ms: not a record
        e.slot_q[1].put("BDONE 1 3 stop 12.0")
        got = e._bdone_after_eos(1, 0.5)
        self.assertEqual(self.payload(got), "BDONE 1 3 stop 12.0",
                         "the end figures must come from the complete record, never the truncated one")

    def test_malformed_then_eof_is_not_a_bdone(self):
        e = self.fixture()
        e.slot_q[1].put("BDONE 1 7 st")                        # mid-record process death
        e.slot_q[1].put(None)
        got = e._bdone_after_eos(1, 0.5)
        self.assertIsNone(self.payload(got), "a truncated record is no BDONE: the EOF state answers")

    def test_foreign_slot_record_not_end(self):
        e = self.fixture()
        e.slot_q[1].put("BDONE 0 3 stop 1.0")                  # bound to 1, field says 0
        got = e._bdone_after_eos(1, 0.3)
        self.assertIsNone(self.payload(got), "a foreign slot's perfect record is not OUR end")

    def test_unicode_digit_number_never_raises_then_eof_answers(self):
        # r03 BLOCKER fixture: '²' passes isdigit() but int() raises - a consumer that lets that
        # escape aborts the caller's finally and leaks the slot busy AND claimed (Codex probe).
        e = self.fixture()
        e.slot_q[1].put("BDONE 1 \u00b2 stop 1.0")             # malformed number BEFORE the EOF
        e.slot_q[1].put(None)
        got = e._bdone_after_eos(1, 0.5)                       # must not raise - EOF answers
        self.assertIsNone(self.payload(got), "malformed number is no BDONE; no exception escapes")
        self.assertTrue(hasattr(server, "_END_EOF") and got[0] == server._END_EOF,
                        "and it is specifically the EOF state (the line was consumed and skipped)")

    def test_oversized_digit_field_never_raises(self):
        e = self.fixture()
        e.slot_q[1].put("BDONE 1 " + "9" * 6000 + " stop 1.0")  # int-conversion limit trap
        e.slot_q[1].put(None)
        got = e._bdone_after_eos(1, 0.5)
        self.assertIsNone(self.payload(got), "4300+ digit fields are malformed, not an exception")
        self.assertTrue(hasattr(server, "_END_EOF") and got[0] == server._END_EOF,
                        "the oversized line was consumed and skipped - the EOF state answers")


class HarnessWire(unittest.TestCase):
    """full real legs over HTTP: corrupt stream lines must not truncate a real request, mint positive
    caches from corrupt records, or fabricate figures."""

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
            ta.start(); tb.start(); ta.join(30); tb.join(30)
        return res

    def _chat(self, text):
        try:
            return self.r.chat(text, 200)
        except Exception as ex:                                # noqa: BLE001 - engine death may surface
            return ("gone", ex)

    def test_garbage_bdones_never_truncate_or_spoof_a_slot(self):
        # corrupt words arrive BEFORE any admission.  Baseline: "BDONE -1" wraps to the LAST slot queue
        # and ends that leg with an empty answer and fabricated figures (generated=3, ms 1.0).
        self.start("--garbage-bdone")
        res = self.two_chats()
        for k in ("a", "b"):
            self.assertIn(k, res, "both requests answered")
            self.assertNotIsInstance(res[k], tuple, f"leg {k} died on the garbage lines")
            self.assertTrue(res[k]["choices"][0]["message"]["content"].strip(),
                            f"leg {k} answered with text, not an empty spoofed end")

    def test_midstream_spoof_bdone_never_ends_or_refills_a_leg(self):
        # a forged "BDONE 1 99 stop 7.0" lands mid-decode in slot 1's stream.  Baseline: that slot's leg
        # ends on the spoof and its usage reports the FORGED figures (generated 99, ms 7).  Fixed: the
        # line fails validation (slot-field/grammar), is ignored, the decode runs on to its real end.
        self.start("--spoof-mid")
        res = self.two_chats()
        for k in ("a", "b"):
            self.assertNotIsInstance(res[k], tuple, "leg survived")
            self.assertEqual(res[k]["usage"]["completion_tokens"], 10,
                             "the fake answers with exactly 10 tokens; a spoofed early end truncates")
        time.sleep(0.3)
        with self.e.slot_cv:                             # baseline: the spoofed leg leaves its REAL lines
            self.assertEqual([list(q.queue) for q in self.e.slot_q], [[], []],
                             "no leftover BT/BDONE residue of a spoofed leg in either slot queue")

    def test_unicode_malformed_at_eos_leaks_neither_busy_nor_claim(self):
        # r03 Codex blocker end-to-end: slot ends at its EOS token; the pipe carries ONE malformed
        # number BDONE ("²") and the engine then stays silent for that slot.  Pre-r03 the helper's
        # int('²') ValueError aborted the request's finally: the slot stayed busy AND claimed with no
        # watcher responsible although the machinery was intact.  Fixed: the helper skips the junk,
        # times out into the untouched A03b arm (BSTOP once, busy+releasing, claim released, held
        # cleared), and a real EOF then closes it conservatively.
        self.start("--unicode-bdone")
        self.helper_err = []
        orig = self.e._bdone_after_eos
        self.e._bdone_after_eos = lambda *a, **kw: self._spy(a, kw, orig)
        res = self.two_chats()
        e = self.e
        with e.slot_cv:
            self.assertEqual(self.helper_err, [], "no exception may escape the read-back")
            armed = set(e.slot_releasing)
            self.assertTrue(armed, "the silent leg handed its slot to the A03b wait")
            self.assertTrue(all(e.slot_busy[s] for s in armed), "busy stands - no early release")
            self.assertEqual(e.slot_claims & armed, set(), "but NO claim leaks beside it")
            self.assertEqual([e.slot_held[s] for s in armed], [[] for _ in armed])
        for s in armed:                                        # end signal - SYNTHETIC here: the fake
            e.slot_q[s].put(None)                              #  lives on, so its pump sends no EOF in
                                                               #  this fixture; the sentinel is the
                                                               #  generic form the watcher consumes
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with e.slot_cv:
                if not any(e.slot_busy[s] for s in armed):
                    break
            time.sleep(0.02)
        with e.slot_cv:
            self.assertFalse(any(e.slot_busy[s] for s in armed), "EOF closes it empty")
            self.assertEqual([e.slot_held[s] for s in armed], [[] for _ in armed])
        self.assertEqual(self.helper_err, [])

    def _spy(self, a, kw, orig):
        try:
            return orig(*a, **kw)
        except Exception as ex:                                # noqa: BLE001 - any escape is a finding
            self.helper_err.append(type(ex).__name__)
            raise

    def test_unicode_malformed_mid_decode_runs_on_to_the_real_end(self):
        # the malformed number word lands MID-decode (after another slot's real BDONE); the decode
        # arm must shrug it off and finish on its own valid record - no exception, real figures.
        # Pre-r03: the decode arm's _apply_bdone int('²') crashed the request thread (HTTPError).
        self.start("--unicode-mid")
        res = self.two_chats()
        for k in ("a", "b"):
            self.assertNotIsInstance(res[k], tuple, "leg survived the malformed word")
            self.assertTrue(res[k]["choices"][0]["message"]["content"].strip(), "full answer")
        time.sleep(0.3)
        with self.e.slot_cv:
            self.assertEqual([list(q.queue) for q in self.e.slot_q], [[], []],
                             "no residue: both legs ended on their own valid records")

    def test_truncated_record_at_eof_never_mints_a_positive_cache(self):
        # the fake dies MID-RECORD after a slot's EOS: an unterminated "BDONE b n st" is the last byte
        # on the pipe.  Baseline: the decode arm treats it as the slot's end and caches a positive tail
        # from caller data although the engine's real state is unknown.  Fixed: the line is ignored, the
        # EOF raises, the finally's arm clears held - every held stays empty, the slot never frees positive.
        self.start("--malformed-bdone")
        self.two_chats()
        e = self.e
        time.sleep(0.4)                                        # let finally/arms settle (no 600 s involved)
        with e.slot_cv:
            self.assertTrue(all(h == [] for h in e.slot_held),
                            f"no positive cache may come from a corrupt record, got {e.slot_held}")


class ValidationMatrix(unittest.TestCase):
    """the classifier itself - field shapes straight from the emitter grammar."""

    CASES = [                                    # (line, valid?)
        ("BDONE 0 7 stop 30.0", True),           # the normal record
        ("BDONE 0 0 length 0.0", True),          # zero figures are valid figures
        ("BDONE 0 7 cancel 1e3", True),          # DELIBERATELY liberal ms spelling: printf %.1f never
                                                #  prints an exponent - accepted harmlessly, not claimed
                                                #  as an emitter form (review nit)
        ("BDONE 0 7 nonfinite 1.0", True),       # #879
        ("BDONE 0 7 stop 1", True),              # complete record at EOF without newline
        ("BDONE 00 7 stop 1.0", False),          # leading zero: printf %d cannot print it
        ("BDONE 0 07 stop 1.0", False),          # ... same for produced (all consumers then agree)
        ("BDONE 0 7 st 1.0", False),             # finish outside the emitter vocabulary
        ("BDONE 0 7 stop", False),               # truncated mid-record
        ("BDONE 0 7 stop nan", False),           # NaN ms: no stats poisoning
        ("BDONE 0 7 stop inf", False),           # inf ms
        ("BDONE 0 7 stop 1e999", False),         # inf by overflow
        ("BDONE 0 7 stop -3", False),            # negative ms
        ("BDONE 0 7 stop abc", False),           # non-numeric ms
        ("BDONE 0 -3 stop 1.0", False),          # negative produced
        ("BDONE -1 7 stop 1.0", False),          # negative slot (wraparound spoof)
        ("BDONE x 7 stop 1.0", False),           # non-numeric slot
        ("BDONE 0 7x stop 1.0", False),          # non-numeric produced
        ("BDONE 0 7 stop 1.0 extra", False),     # trailing junk: unknown grammar
        ("BDONE 0 7 STOP 1.0", False),           # case-sensitive emitter vocabulary
        ("BDONE 0 \u00b2 stop 1.0", False),      # r03: SUPERSCRIPT TWO passes isdigit() - int() would
        ("BDONE \u00b2 7 stop 1.0", False),      #  raise ValueError; the classifier returns False
        ("BDONE 0 \u0667 stop 1.0", False),      # Arabic-Indic seven: int('٧') PARSES (7!) - no raise,
                                                 #  a fabricated figure; the ASCII gate rejects it
        ("BDONE 0 123456789012345678901234567890\u00b2 stop 1.0", False),
        ("BDONE 0 " + "9" * 6000 + " stop 1.0", False),   # beyond CPython's 4300-digit int limit:
        ("BDONE 0 7 stop \u0667.0", False),      # ... and an ASCII gate on ms: float() would read
                                                 #  Arabic-Indic digits as figures ('٧.٠' -> 7.0)
    ]

    def test_classifier(self):
        for line, ok in self.CASES:
            self.assertIs(server.StrataEngine._bdone_valid(line.split()), ok, line)

    def test_bound_slot_must_match(self):
        self.assertTrue(server.StrataEngine._bdone_valid("BDONE 2 7 stop 1.0".split(), slot=2))
        self.assertFalse(server.StrataEngine._bdone_valid("BDONE 2 7 stop 1.0".split(), slot=0))

    def note_split(self):
        # split() collapses repeated spaces: "BDONE  0 ..." would still parse as five fields.  The
        # emitter's printf can never print double spaces, so field-wise split() stays; marked as a
        # decision, not a defect.
        pass
