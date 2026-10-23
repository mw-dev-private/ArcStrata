"""A03d - a consumed EOF must reach the cleanup as a CONFIRMED end, not as a silent timeout.

On the parent, `_bdone_after_eos` returned None for BOTH a quiet queue and its own pump EOF.  A request
whose engine died right after the EOS token therefore armed a follow-up BDONE watcher on a queue whose
EOF had already been eaten - the watcher polls for a signal that no longer exists and only lands in the
600 s quarantine.  The candidate distinguishes the four states and closes a confirmed EOF directly with
an empty cache (no follower, no BSTOP), while the timeout arm keeps the whole A03b contract.

Harness fixtures are opt-in FAKE_BATCH flags: `--die-after-eos` (the fake process exits AT a slot's EOS,
so its pump really sends EOF instead of the BDONE) and `--silent-after-eos` (process lives, the slot's
BDONE never comes - the timeout arm).  Pump-EOF means that engine's end; general availability still
hangs on the lifecycle - nothing here claims a production outage.

Parent control (parent server.py + these test files over the parent archive): the first test goes red by
BEHAVIOUR (slot stays busy+releasing on a dead queue); the control arm is green on both trees and turns
red only under the "timeout as EOF" mutation; the swap test is a candidate-vs-guard-mutation pin.
"""
import queue

import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.test_parallel as tp
from serve import server


class EofHandoff(unittest.TestCase):
    """Full real legs: StrataEngine + Service over HTTP + the fake engine."""

    def start_engine(self, flag):
        self.r = tp.ParallelService()
        self.r.start(2, slot_cache=True, reuse=True, more=("--readfield", flag))
        self.e = e = self.r.engine
        self.sent = []
        real_send = e._send
        e._send = lambda line: (self.sent.append(line), real_send(line))[1]   # count every control word

    def run_two_slot_requests(self, wrapper=None):
        e = self.e
        self.calls = []
        orig = e._bdone_after_eos

        def spy(slot, *a, **kw):                     # always record: no test may pass vacuously
            if wrapper is not None:                  # (the wrapper itself appends inside)
                return wrapper(slot, a, kw, orig, self.calls)
            r = orig(slot, *a, **kw)
            self.calls.append((slot, r))
            return r
        e._bdone_after_eos = spy
        res = {}
        with mock.patch.object(server, "EOS_IDS", {257}):
            ta = threading.Thread(target=lambda: res.setdefault("a", self._chat_safe("List three primes please")))
            tb = threading.Thread(target=lambda: res.setdefault("b", self._chat_safe("Name two rivers in Germany")))
            ta.start(); tb.start(); ta.join(30); tb.join(30)
        return res

    def _chat_safe(self, text):
        try:
            return self.r.chat(text, 200)
        except Exception as ex:                       # noqa: BLE001 - engine death may surface as 5xx
            return ("gone", ex)

    def tearDown(self):
        if getattr(self, "r", None):
            self.r.tearDown()

    def _swap_now(self, batch=3):
        """restart()'s atomic step (see A03c): all arrays + gen together under slot_cv, successor with
        distinctive queues/claims/held so ANY old-generation write is visible."""
        e = self.e
        with e.slot_cv:
            e.batch = batch
            e.slot_q = [queue.Queue() for _ in range(batch)]
            for s in range(batch):
                e.slot_q[s].put(f"BT {s} 777")
            e.slot_busy = [True] * batch
            e.slot_claims = set(range(batch))
            e.slot_releasing, e.slot_quarantine = set(), set()
            e.slot_held = [[900 + s] for s in range(batch)]
            e.slot_used = [0.0] * batch
            e.slot_live = [None] * batch
            e.slot_order = list(range(batch))
            e.slot_group, e.slot_groups, e.slot_gs = [0] * batch, 1, batch
            e._orphan_seen, e._ctl_erred, e._yielded = {}, False, None
            e.gen += 1

    # ---- the reproduced defect: EOF after the EOS token must NOT leave a follower polling a dead queue

    def test_confirmed_eof_closes_the_slot_no_follower_polls_a_dead_queue(self):
        # the fake engine dies AT the EOS token: the slot's consumer reads exactly ONE EOF from its own
        # queue.  Parent: it arms a BDONE watcher anyway -> that slot stays busy + releasing forever
        # (a real run would poll it into the 600 s quarantine); candidate: closes with empty held, no
        # follower, no BSTOP for the EOF-closed leg.
        self.start_engine("--die-after-eos")
        self.run_two_slot_requests()
        e = self.e
        self.assertTrue(self.calls, "the slot legs reached the read-back at all (else every assertion "
                                    "below would be vacuous on a fixture that died too early)")
        if hasattr(server, "_END_EOF"):                # candidate-only: the EOF state itself must occur
            self.assertTrue(any(r[0] == server._END_EOF for _, r in self.calls),
                            "a leg really consumed its queue's EOF (the case under test)")
        end, bad = time.monotonic() + 3.0, None
        while time.monotonic() < end:                  # settle: the candidate closes within milliseconds
            with e.slot_cv:
                bad = set(e.slot_releasing) or any(e.slot_busy)
            if not bad:
                break
            time.sleep(0.05)
        self.assertFalse(bad, "a watcher was left armed on a queue whose EOF was already consumed "
                              "(parent behaviour: busy+releasing until the 600 s quarantine)")
        self.assertEqual(set(e.slot_releasing), set())
        self.assertFalse(any(e.slot_busy))
        self.assertEqual(set(e.slot_quarantine), set())
        self.assertEqual(e.slot_held, [[], []], "EOF invents no cache")
        self.assertLessEqual(sum(1 for x in self.sent if x.startswith("BSTOP")), 1,
                             "the EOF-closed leg never BSTOPs a dead engine (at most the OTHER leg arms once)")

    # ---- control arm: quiet (timeout) stays the locked A03b path - red under "timeout treated as EOF"

    def test_timeout_arm_stays_locked_and_the_reaper_skips_it(self):
        # engine alive, no BDONE after the EOS: the helper times out -> the A03b path must hold the slot
        # busy+releasing unpickably, survive the orphan reaper, and close only on a real end signal.
        self.start_engine("--silent-after-eos")
        self.run_two_slot_requests()
        e = self.e
        with e.slot_cv:
            armed = set(e.slot_releasing)
            self.assertTrue(armed, "the quiet legs handed their slots to the BDONE wait (A03b contract)")
            self.assertTrue(all(e.slot_busy[s] for s in armed), "locked busy, unpickable")
        old_orphan = e.ORPHAN_SLOT_S
        e.ORPHAN_SLOT_S = 0.2
        try:
            time.sleep(0.5)
            with e.slot_cv:
                self.assertEqual(e._reap_orphan_slots(), [], "the reaper may not free a releasing slot")
                self.assertTrue(all(e.slot_busy[s] for s in armed))
        finally:
            e.ORPHAN_SLOT_S = old_orphan
        with e.slot_cv:                                # deliver an end signal - SYNTHETIC here: the fake
            for s in armed:                            #  stays alive under --silent-after-eos, so a real
                e.slot_q[s].put(None)                  #  pump EOF would not occur in this fixture; the
                                                       #  watcher treats the sentinel generically as THE end
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with e.slot_cv:
                if not any(e.slot_busy[s] for s in armed):
                    break
            time.sleep(0.05)
        with e.slot_cv:
            self.assertFalse(any(e.slot_busy[s] for s in armed), "the confirmed EOF closes it after all")
            self.assertEqual(set(e.slot_releasing), set())
            self.assertEqual([e.slot_held[s] for s in armed], [[] for _ in armed], "watcher path: empty cache")

    # ---- generation swap between the EOF read and the checked close: successor arrays untouched

    def eof_close_line(self):
        """line number of the `with self.slot_cv:` directly inside the eof_in close branch."""
        lines = Path(server.__file__).read_text().splitlines()
        anchor = next(i for i, l in enumerate(lines) if 'elif phase == "slot" and eof_in:' in l)
        for j in range(anchor + 1, anchor + 4):
            if "with self.slot_cv:" in lines[j]:
                return j + 1                                  # 1-based, as frame.f_lineno
        raise AssertionError("eof-close lock line not found")

    def mk_injection_trace(self, batch):
        """Codex r02 finding: the injection that matters lands BETWEEN the two lock sections - the
        outer skip gate has already passed, restart()'s atomicity cannot help (it excludes only
        mid-section entry), and only the re-check inside the eof-close lock can stop an old leg from
        rewriting the successor.  Deterministic: threading.settrace BEFORE the request threads start
        (a frame is only traced if the tracer predates the frame), firing exactly once at the
        eof-close `with` line."""
        target, self.inj = self.eof_close_line(), []

        def trace(frame, event, arg):
            if (event == "line" and frame.f_code.co_filename == server.__file__
                    and frame.f_lineno == target and frame.f_locals.get("self") is self.e
                    and not self.inj):
                self.inj.append(target)
                self._swap_now(batch=batch)                   # atomic successor, under slot_cv
                return None                                   # one shot: stop tracing this frame
            return trace
        return trace

    def assert_successor_intact(self, batch):
        time.sleep(0.3)                                       # let any wrong write attempt land
        with self.e.slot_cv:
            self.assertEqual([list(q.queue) for q in self.e.slot_q],
                             [["BT %d 777" % s] for s in range(batch)], "successor queues intact")
            self.assertEqual(self.e.slot_held, [[900 + s] for s in range(batch)], "successor cache intact")
            self.assertEqual(self.e.slot_claims, set(range(batch)), "successor claims intact")
            self.assertTrue(all(self.e.slot_busy), "successor busy flags intact")

    def test_swap_between_the_lock_sections_is_caught_by_the_inner_check(self):
        # r03 correction (Codex race probe): injection AFTER the outer gate, right at the eof-close
        # lock - the inner re-check is the ONLY guard there.  Successor keeps enough slots so the
        # mutation comparison sees the WRONG WRITE, not an IndexError.
        if not hasattr(server, "_END_EOF"):
            self.skipTest("_END_EOF is candidate-only; on the parent this pin is a shape control")
        self.start_engine("--die-after-eos")
        threading.settrace(self.mk_injection_trace(batch=3))
        try:
            self.run_two_slot_requests()
        finally:
            threading.settrace(None)
        self.assertTrue(self.inj, "the injection really reached the eof-close lock (else vacuous)")
        self.assert_successor_intact(3)

    def test_shrunken_successor_between_the_lock_sections_is_survived(self):
        # same injection but the successor SHRANK to one slot: the candidate skips every write and
        # raises nothing; this pins the skip path itself (mutation detection rides on the test above).
        if not hasattr(server, "_END_EOF"):
            self.skipTest("_END_EOF is candidate-only; on the parent this pin is a shape control")
        self.start_engine("--die-after-eos")
        threading.settrace(self.mk_injection_trace(batch=1))
        try:
            self.run_two_slot_requests()
        finally:
            threading.settrace(None)
        self.assertTrue(self.inj, "the injection really reached the eof-close lock (else vacuous)")
        self.assert_successor_intact(1)

    def test_swap_after_the_eof_read_never_closes_the_successors_slot(self):
        # the swap lands right after the helper consumed OUR EOF (_END_EOF) but BEFORE the outer skip
        # gate is evaluated: that gate dominates this timing (r03 correction: this pin does NOT probe
        # the inner re-check - the test above injects after the gate and does).  Candidate-only by
        # nature: the parent cannot express the state - a shape control plus end-to-end successor pin.
        if not hasattr(server, "_END_EOF"):
            self.skipTest("_END_EOF is candidate-only; on the parent this pin is a shape control")
        swapped = []

        def wrapper(slot, a, kw, orig, calls):
            r = orig(slot, *a, **kw)
            calls.append((slot, r))
            if not swapped and r[0] == server._END_EOF:
                swapped.append(1)
                self._swap_now()
            return r

        self.start_engine("--die-after-eos")
        self.run_two_slot_requests(wrapper=wrapper)
        e = self.e
        self.assertTrue(swapped, "an _END_EOF read actually happened (else this pin would be vacuous)")
        time.sleep(0.3)                                # let any wrong follower/close attempt land
        with e.slot_cv:
            self.assertEqual([list(q.queue) for q in e.slot_q], [["BT %d 777" % s] for s in range(3)],
                             "the successor's queues stay intact")
            self.assertEqual(e.slot_held, [[900 + s] for s in range(3)], "and its cache too")
            self.assertEqual(e.slot_claims, {0, 1, 2})
            self.assertTrue(all(e.slot_busy))
        self.assertTrue(any(r[0] == server._END_EOF for _, r in self.calls))
