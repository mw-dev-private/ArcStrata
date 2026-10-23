"""A03c - the EOS read-back (#1869) may only touch ITS OWN engine generation.

`_bdone_after_eos` used to read the live `self.slot_q[slot]` after the caller's UNLOCKED gen check, and
`_apply_bdone` wrote the figures without re-checking: a restart() swapping in between (arrays+gen move as
ONE step under slot_cv since A03b r02) made the old leg consume the successor generation's BT/BDONE lines,
or - when the successor has FEWER slots - hit an IndexError right inside the request finally, aborting the
rest of that finally chain (control-line release, counters).  With batch on, `self.last` is THREAD-LOCAL
(the engine's property), so cross-thread stats poisoning was NOT a real channel - fixed scope: bound
queue, generation gates, one checked apply step.

Race tests use barrier queues and a manual swap that mirrors restart()'s atomic step faithfully (all
arrays AND gen together UNDER slot_cv - stricter than the older unlocked manual swaps in quiet_cancel/
A03b tests).  Four of the five go red on the parent archive 73ad57d5 by BEHAVIOUR (successor drained x2,
apply after an ended generation, IndexError into a shrunken array); the gate-before-bind test is red there
by SIGNATURE (`born=` is unknown to the parent helper) - unavoidable, because without `born` the candidate
would bind the successor too.  The parity tests (healthy BDONE, EOF, timeout/Empty, nonfinite pass-through,
index guard) are positive controls: green on parent and candidate alike.
"""
import queue
import threading
import unittest
from unittest import mock

from serve import server


class BarrierQ(queue.Queue):
    """A real queue.Queue that fires a one-shot hook at the start of its next get() - the barrier for
    'the restart swaps right here, while the read-back waits'."""

    def __init__(self):
        super().__init__()
        self.hook = None

    def get(self, timeout=None):
        h, self.hook = self.hook, None
        if h is not None:
            h()
        return super().get(timeout=timeout)


SENTINEL = {"generated": 111, "finish": "sentinel", "decode_ms": 42.0}


def swap_in(e, batch, lines_by_slot, new_last=None):
    """restart()'s atomic step (A03b r02): every slot array, the ownership sets AND gen change together
    under slot_cv.  The successor's slots get distinctive lines so ANY consumption is visible."""
    with e.slot_cv:
        e.batch = batch
        e.slot_q = [queue.Queue() for _ in range(batch)]
        for s, lines in lines_by_slot.items():
            for line in lines:
                e.slot_q[s].put(line)
        e.slot_busy = [True] * batch
        e.slot_claims = set(lines_by_slot)
        e.slot_releasing, e.slot_quarantine = set(), set()
        e.slot_held = [[] for _ in range(batch)]
        e.slot_used = [0.0] * batch
        e.slot_live = [None] * batch
        e.slot_order = list(range(batch))
        e.slot_group = [0] * batch                      # fidelity: the real swap replaces these too
        e.slot_groups, e.slot_gs = 1, batch
        e._orphan_seen, e._ctl_erred, e._yielded = {}, False, None
        e.gen += 1
        if new_last is not None:
            e.last = new_last


class EosHarness(unittest.TestCase):
    """Direct helper calls on a minimal engine: real queues, real (short) waits, injected swaps."""

    def fixture(self, batch=3):
        e = object.__new__(server.StrataEngine)
        e.gen, e.batch = 0, batch
        self.qs = [BarrierQ() for _ in range(batch)]
        e.slot_q = list(self.qs)
        e.slot_busy = [False] * batch
        e.slot_claims, e.slot_releasing, e.slot_quarantine = {1}, set(), set()
        e.slot_held, e.slot_used, e.slot_live = [[] for _ in range(batch)], [0.0] * batch, [None] * batch
        e.slot_order, e._orphan_seen = list(range(batch)), {}
        e.slot_cv = threading.Condition()
        e.last = {"generated": 7, "finish": "length", "decode_ms": 7.0}
        return e

    # ---- the three race windows Codex named; same call shape on both trees (born= trailing keyword)

    def test_swap_between_caller_check_and_helper_binds_nothing(self):
        # the caller's unlocked `self.gen == born` passed, THEN the swap landed, THEN the helper ran
        e = self.fixture()
        succ = ["BT 1 777", "BDONE 1 999 stop 99.0"]
        swap_in(e, 3, {1: list(succ)}, dict(SENTINEL))
        got = e._bdone_after_eos(1, 0.3, born=0)            # the request was born on generation 0
        self.assertEqual(got, (server._END_GONE, None), "a gone generation reads nothing, not even a line that looks like BDONE")
        self.assertEqual(list(e.slot_q[1].queue), succ, "the successor's lines stay in the successor's queue")
        self.assertEqual(e.last, SENTINEL, "and the helper leaves the successor's figures alone")

    def test_swap_during_the_wait_stays_on_the_old_queue(self):
        # helper bound the old queue and waits; the swap lands while its FIRST get returns a line of ours
        e = self.fixture()
        succ = ["BT 1 777", "BDONE 1 999 stop 99.0"]
        self.qs[1].put("BT 1 65")                          # one of OUR tokens: consumed, loop continues
        self.qs[1].hook = lambda: swap_in(e, 3, {1: list(succ)}, dict(SENTINEL))
        got = e._bdone_after_eos(1, 0.3)                   # no born: current-at-entry (0) is used
        self.assertEqual(got, (server._END_NONE, None), "the old queue said Empty after our BT; the successor is none of its business")
        self.assertEqual(list(e.slot_q[1].queue), succ, "not one successor line consumed")
        self.assertEqual(list(self.qs[1].queue), [], "our own queue is ours to drain")

    def test_fewer_successor_slots_never_index_the_new_arrays(self):
        e = self.fixture(batch=3)
        self.qs[1].put("BT 1 65")
        self.qs[1].hook = lambda: swap_in(e, 1, {0: ["BDONE 0 555 stop 5.0"]}, dict(SENTINEL))
        got = e._bdone_after_eos(1, 0.3)                   # slot 1 does not exist on the successor
        self.assertEqual(got, (server._END_NONE, None), "no IndexError into a shrunken array (the parent crashed right here)")
        self.assertEqual(list(e.slot_q[0].queue), ["BDONE 0 555 stop 5.0"])

    # ---- parity controls: healthy contract untouched (green on parent and candidate)

    def test_healthy_readback_still_returns_the_slot_bdone(self):
        e = self.fixture()
        for line in ("BT 1 65", "BDONE 1 12 stop 30.0"):
            self.qs[1].put(line)
        self.assertEqual(e._bdone_after_eos(1, 0.5), (server._END_BDONE, "BDONE 1 12 stop 30.0"))
        self.assertTrue(self.qs[1].empty(), "own lines may be consumed")

    def test_eof_on_our_queue_ends_the_readback(self):
        e = self.fixture()
        self.qs[1].put(None)
        self.assertEqual(e._bdone_after_eos(1, 0.5), (server._END_EOF, None),
                         "A03d: the consumed EOF is now a CONFIRMED end state, not a silent None")

    def test_quiet_slot_times_out_like_always(self):
        e = self.fixture()
        self.assertEqual(e._bdone_after_eos(1, 0.2), (server._END_NONE, None))

    def test_nonfinite_bdone_passes_through_for_the_caller(self):
        e = self.fixture()
        self.qs[1].put("BDONE 1 0 nonfinite 1.0")
        self.assertEqual(e._bdone_after_eos(1, 0.5), (server._END_BDONE, "BDONE 1 0 nonfinite 1.0"))

    def test_slot_gone_in_our_generation_returns_gone_without_waiting(self):
        e = self.fixture()
        e.slot_q = []                                     # same-gen shrink: only reachable via odd states,
        self.assertEqual(e._bdone_after_eos(1, 5.0), (server._END_GONE, None))   #  the len guard answers instantly either way


class EosCallerRace(unittest.TestCase):
    """The REAL finally of generate_batched over the fake-engine harness (test_parallel's #1869 setup):
    the swap barrier sits around the helper call itself, so check-then-act races land exactly in the
    caller's window.  Discriminators: the successor generation's queues (parent re-reads the live slot_q
    and DRAINS them) and a spy on `_apply_bdone` recording the generation live at every hand-over."""

    def start_two_slot_requests(self, wrapper, spy_apply=False):
        import contextlib
        import serve.test_parallel as tp

        class Runner(tp.ParallelService):
            def runTest(self):
                pass

        self.r = Runner()
        self.r.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        e = self.engine = self.r.engine
        orig = e._bdone_after_eos
        self.calls, self.apply_gens = [], []

        def wrapped(slot, *a, **kw):
            return wrapper(slot, a, kw, orig, self.calls)

        def apply_spy(f):
            def inner(ff):
                self.apply_gens.append(e.gen)           # which generation is live when the caller hands
                return f(ff)                            #  the figures over?
            return inner

        patches = [mock.patch.object(server, "EOS_IDS", {257}), mock.patch.object(e, "_bdone_after_eos", wrapped)]
        if spy_apply:
            patches.append(mock.patch.object(e, "_apply_bdone", apply_spy(e._apply_bdone)))
        with contextlib.ExitStack() as stack:
            for pt in patches:
                stack.enter_context(pt)
            res = {}
            ta = threading.Thread(target=lambda: res.setdefault("a", self.r.chat("List three primes please", 200)))
            tb = threading.Thread(target=lambda: res.setdefault("b", self.r.chat("Name two rivers in Germany", 200)))
            ta.start(); tb.start(); ta.join(30); tb.join(30)
        # capture the observation BEFORE tearDown: unload() resets state, and queues too
        self.queues_after = [list(q.queue) for q in e.slot_q]
        self.r.tearDown()

    def _swap_now(self, batch=3):
        swap_in(self.engine, batch, {0: ["BT 0 777", "BDONE 0 999 stop 99.0"],
                                     1: ["BT 1 777", "BDONE 1 999 stop 99.0"]}, dict(SENTINEL))

    def test_swap_after_the_readback_applies_only_within_its_generation(self):
        # swap lands right after the helper returned OUR real BDONE: the parent (unchecked) applies the
        # figures of a generation that just ENDED - red; the candidate's checked step drops them instead.
        fired = []

        def wrapper(slot, a, kw, orig, calls):
            r = orig(slot, *a, **kw)
            if not fired:
                fired.append(1)
                self._swap_now()
            calls.append(r)
            return r

        self.start_two_slot_requests(wrapper, spy_apply=True)
        self.assertTrue(any(r[0] == server._END_BDONE for r in self.calls),
                        "the read-back itself worked (healthy #1869 leg, our own BDONE, before the swap)")
        self.assertEqual(self.apply_gens, [], "figures of an ended generation are dropped, never applied")
        for s in (0, 1):
            self.assertEqual(self.queues_after[s], [f"BT {s} 777", f"BDONE {s} 999 stop 99.0"],
                             "the successor's queues stay untouched either way")

    def test_swap_between_gate_and_helper_consumes_nothing(self):
        # the caller's gate passed; the swap lands BEFORE the helper binds: parent re-reads live slot_q
        # and DRAINS the successor generation's queue (that conversation loses the BT/BDONE lines that
        # are ITS OWN #1869 read-back material).  Red on the parent by behaviour.
        fired = []

        def wrapper(slot, a, kw, orig, calls):
            if not fired:
                fired.append(1)
                self._swap_now()
            calls.append(r := orig(slot, *a, **kw))
            return r

        self.start_two_slot_requests(wrapper)
        self.assertTrue(self.calls, "the read-back ran (else this test would be vacuous)")
        for s in (0, 1):
            self.assertEqual(self.queues_after[s], [f"BT {s} 777", f"BDONE {s} 999 stop 99.0"],
                             "no successor line consumed")


