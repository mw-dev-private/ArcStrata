"""A03b - the 600 s BDONE deadline is not an end.  _release_slot_when_done must quarantine a slot whose engine
stayed silent instead of freeing it, and only a confirmed end (BDONE, nonfinite BDONE, EOF, generation end) may
release it.  Virtual clock + scripted fake queues: exact states, exact BSTOP counts, no real 600 s.

Every "hold" test is a positive control: run this file against the parent archive (f0cbd2bd) and the pre-fix
fall-through fails them (wrong release, BSTOP twice, watcher outlives its generation).
"""
import queue
import threading
import time
import unittest
from collections import deque
from unittest import mock

from serve import server

_REAL_MONO = time.monotonic          # captured before any patch: wall-clock waits while monotonic is virtual


def until(cond, what, timeout=5.0):
    t0 = _REAL_MONO()
    while not cond():
        if _REAL_MONO() - t0 > timeout:
            raise AssertionError(f"timeout waiting for {what}")
        time.sleep(0.005)


class FakeQ:
    """Queue for a watcher under a virtual clock.  While now < fast_until, get() advances the virtual clock and
    says Empty (fast-forwarding the quiet minutes); afterwards it waits for scripted lines without moving time."""

    def __init__(self, now, lines=(), fast_until=1e18):
        self.now, self.lines, self.cv = now, deque(lines), threading.Condition()
        self.fast_until, self.gets = fast_until, 0

    def put(self, line):
        with self.cv:
            self.lines.append(line)
            self.cv.notify_all()

    def get(self, timeout):
        self.gets += 1
        with self.cv:
            while not self.lines:
                if self.now[0] < self.fast_until:
                    self.now[0] += timeout
                    raise queue.Empty
                self.cv.wait(min(timeout, 0.02))         # bounded like a real Queue.get(timeout=...): the
                if not self.lines:                       #  watcher wakes at its loop top and can notice the
                    raise queue.Empty                    #  generation change instead of hanging in the fake
            return self.lines.popleft()


class CleanupHarness(unittest.TestCase):
    def fixture(self, batch=3, fast_until=1e18):
        self.now = [0.0]
        e = self.engine = object.__new__(server.StrataEngine)
        e.gen = 0
        e.slot_q = [None] * batch
        e.slot_busy = [False] * batch
        e.slot_claims, e.slot_releasing, e.slot_quarantine = set(), set(), set()
        e.slot_held = [[] for _ in range(batch)]
        e.slot_used, e.slot_live, e._orphan_seen = [0.0] * batch, [None] * batch, {}
        e.slot_order = list(range(batch))
        e.slot_cv = threading.Condition()
        self.sent = []
        e._send = self.sent.append
        self.q = FakeQ(self.now, fast_until=fast_until)
        e.slot_q[1] = self.q
        e.slot_busy[1], e.slot_claims = [b == 1 for b in range(batch)], {1}
        return e

    def state(self):
        e = self.engine
        return {"busy": list(e.slot_busy), "releasing": sorted(e.slot_releasing),
                "quarantine": sorted(e.slot_quarantine), "held": list(e.slot_held)}

    # ---- the contract's core: deadline passes, engine alive and silent -> quarantine, never a release ----

    def test_silent_live_engine_past_the_deadline_stays_quarantined(self):
        e = self.fixture(fast_until=610.0)
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            until(lambda: 1 in e.slot_quarantine, "quarantine at the deadline")
            self.assertEqual(self.sent, ["BSTOP 1"])            # exactly one BSTOP
            self.assertTrue(e.slot_busy[1], "the deadline must not free a possibly-busy slot")
            self.assertIn(1, e.slot_releasing, "the watcher stays responsible (no silent flag loss)")
            self.assertEqual(e.slot_held[1], [], "a deadline invents no cache state")
            with e.slot_cv:                                     # pick never sees it as free
                self.assertNotEqual(e.pick_slot([7, 8]), 1, "a quarantined slot must not be pickable")
            for _ in range(3):                                  # orphan reaper over its whole window: no touch
                with self.q.cv:                                 # shared virtual clock: advance it under the
                    self.now[0] += server.StrataEngine.ORPHAN_SLOT_S + 1   #  fake queue's lock too (no races)
                with e.slot_cv:
                    self.assertEqual(e._reap_orphan_slots(), [], "quarantine is not an orphan")
            self.assertTrue(e.slot_busy[1])
            self.assertEqual(self.state()["quarantine"], [1])
            self.q.put(None)                                 # wind-down that doubles as an assertion: the watcher
            until(lambda: not e.slot_busy[1], "EOF resolves the quarantine")  # stayed responsible until confirmed
            self.assertEqual(self.state()["quarantine"], [], "quarantine cleared with the confirmed end")
            self.assertEqual(self.state()["releasing"], [])

    def test_late_bdone_after_the_deadline_releases_with_empty_cache(self):
        e = self.fixture(fast_until=610.0)
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            until(lambda: 1 in e.slot_quarantine, "quarantine")
            self.q.put("BT 1 65")                               # the engine finally speaks: drain stays bounded
            self.q.put("BDONE 1 1 stop 1.0")                    # ... and confirms its end
            until(lambda: not e.slot_busy[1], "confirmed release")
            self.assertEqual(e.slot_held[1], [], "quarantine gave cache retention up: no late positive cache")
            self.assertEqual(self.state()["quarantine"], [], "quarantine cleared together with the release")
            self.assertEqual(self.state()["releasing"], [])

    def test_bt_lines_behind_the_deadline_are_drained_not_retained(self):
        e = self.fixture(fast_until=610.0)
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            until(lambda: 1 in e.slot_quarantine, "quarantine")
            got = self.q.gets
            for t in range(50):                                 # a chatty engine must not grow anything
                self.q.put(f"BT 1 {200 + t}")
            until(lambda: self.q.gets > got + 50, "drain of all 50 lines")
            self.assertEqual(e.slot_held[1], [])
            self.q.put("BDONE 1 51 stop 1.0")
            until(lambda: not e.slot_busy[1], "release")
            self.assertEqual(e.slot_held[1], [], "no unbounded tail retention behind the deadline")

    # ---- the confirmed ends keep their parent-identical semantics ----

    def test_bdone_before_the_deadline_finishes_the_cache_as_before(self):
        e = self.fixture()
        self.q = self.engine.slot_q[1] = FakeQ(self.now, lines=["BT 1 65", "BT 1 66", "BDONE 1 2 stop 1.0"])
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            until(lambda: not e.slot_busy[1], "release")
            self.assertEqual(e.slot_held[1], [104, 105, 65], "stream + BT minus the last token, unchanged")
            self.assertEqual(self.sent, ["BSTOP 1"])

    def test_nonfinite_bdone_releases_without_positive_cache(self):
        e = self.fixture()
        self.engine.slot_q[1] = FakeQ(self.now, lines=["BDONE 1 0 nonfinite 1.0"])
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            until(lambda: not e.slot_busy[1], "release")
            self.assertEqual(e.slot_held[1], [])

    def test_eof_on_the_slot_queue_is_a_confirmed_end(self):
        e = self.fixture()
        self.engine.slot_q[1] = FakeQ(self.now, lines=[None])
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            until(lambda: not e.slot_busy[1], "release")
            self.assertEqual(e.slot_held[1], [])
            self.assertEqual(self.state()["releasing"], [])

    # ---- generations ----

    def test_cleanup_of_a_vanished_generation_never_arms_on_the_successor(self):
        # A03b r02 (review finding 3): the consumer's finally raced a restart - with the swap now atomic under
        # slot_cv, the entry's born re-check must refuse the stale cleanup before mark, BSTOP or binding.
        e = self.fixture()
        succ_q = FakeQ(self.now, lines=["BT 1 999", "BDONE 1 1 stop 1.0"])
        e.gen = 1                                        # the restart fully replaced state...
        e.slot_q = [None, succ_q, None]
        e.slot_busy = [False, True, False]               # ...and the successor is decoding in slot 1 right now
        e.slot_claims, e.slot_releasing, e.slot_quarantine = {1}, set(), set()
        e.slot_held = [[], [104, 105], []]
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105], born=0)   # the OLD generation's consumer exit arrives
        self.assertEqual(self.sent, [], "no BSTOP may reach the successor's engine")
        self.assertEqual(e.slot_releasing, set(), "no releasing mark lands in the successor's set")
        self.assertEqual(succ_q.gets, 0, "the successor's queue is never drained by a stranger")
        self.assertEqual(e.slot_held[1], [104, 105], "the successor's cache is untouched")
        self.assertTrue(e.slot_busy[1], "the successor keeps its slot")

    def test_a_defaulted_cleanup_never_indexes_the_smaller_successor(self):
        # the entry's index guard re-run INSIDE the lock (r03, review r2 nit 4): with born defaulting to the
        # current gen, an old-gen index must still be refused against a successor that came up with fewer slots.
        e = self.fixture(batch=3)
        e.gen = 1
        e.slot_q, e.slot_busy = [None, None], [False, True]
        e.slot_claims, e.slot_releasing, e.slot_quarantine = {1}, set(), set()
        e.slot_held, e.slot_used, e.slot_live, e.slot_order = [[], []], [0.0] * 2, [None] * 2, [0, 1]
        e._release_slot_when_done(2, [104, 105])          # index 2 belongs to the dead generation
        self.assertEqual(self.sent, [], "no BSTOP for a vanished slot")
        self.assertEqual(e.slot_releasing, set(), "no mark lands in the successor's set")

    def test_watcher_error_keeps_the_lock_then_one_replacement_resolves(self):
        # A03b r04 (Codex P1): a watcher error is not a confirmed end - the slot goes to the quarantine lock
        # (never to the orphan reaper) and EXACTLY ONE replacement watcher takes responsibility; when that one
        # confirms the end, everything clears normally.
        e = self.fixture()
        state = {"boom": 1}

        def flaky(timeout):
            self.now[0] += 5.0
            if state["boom"]:
                state["boom"] -= 1
                raise RuntimeError("simulated watcher bug")
            return "BDONE 1 1 stop 1.0"

        self.q.get = flaky
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105], born=0)
            until(lambda: not e.slot_busy[1], "replacement watcher confirmed the end")
        self.assertEqual(self.sent, ["BSTOP 1"], "the replacement must not BSTOP again")
        self.assertEqual(e.slot_releasing, set(), "all marks cleared with the confirmed end")
        self.assertEqual(e.slot_quarantine, set())
        self.assertEqual(e.slot_held[1], [], "A03b r06: a replacement-confirmed end closes conservatively")

    # ---- A03b r06 (Codex codex-A03b-289422be-review): the replacement must never rebuild a LIEB fragment

    def _stepped_get(self, steps):
        now, calls = self.now, []

        def get(timeout):
            step = steps[len(calls)]
            adv, what = step if isinstance(step, tuple) else (0, step)   # plain line = no clock advance
            calls.append(1)
            now[0] += adv
            if what == "!":
                raise RuntimeError("injected watcher bug")
            if what is None:
                raise queue.Empty
            return what
        return get, calls

    def test_replacement_never_replays_history_lost_at_the_crash(self):
        # Codex' probe, CPU-side: BT 65,66 read by watcher one, crash, replacement reads BT 67,68, BDONE -
        # the real history is 104,105,65,66,67,[68]; a rebuilt tail would claim [104,105,67] (not a prefix).
        # Rule: after ANY watcher failure the replacement closes with held=[] - never positive on fragments.
        e = self.fixture()
        get, calls = self._stepped_get(["BT 1 65", "BT 1 66", "!", "BT 1 67", "BT 1 68", "BDONE 1 4 stop 1.0"])
        self.q.get = get
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105], born=0)
            until(lambda: not e.slot_busy[1], "replacement confirms before the deadline")
        self.assertEqual(e.slot_held[1], [], "no positive cache from a partial history")
        self.assertEqual(self.sent, ["BSTOP 1"], "one BSTOP for two watchers")
        self.assertEqual(len(calls), 6, "both watchers drained, none replayed the stream")
        self.assertEqual(e.slot_quarantine, set(), "the confirmed end still clears the lock")

    def test_failure_after_the_last_bt_closes_empty(self):
        e = self.fixture()
        get, _ = self._stepped_get(["BT 1 65", "BT 1 66", "!", "BDONE 1 3 stop 1.0"])
        self.q.get = get
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105], born=0)
            until(lambda: not e.slot_busy[1], "replacement confirms")
        self.assertEqual(e.slot_held[1], [])
        self.assertEqual(self.sent, ["BSTOP 1"], "one BSTOP for two watchers")
        self.assertEqual(e.slot_quarantine, set(), "the confirmed end clears the lock")

    def test_healthy_watcher_still_keeps_the_full_prefix(self):
        # the control arm: same BT/BDONE lines, NO watcher failure - parent-parity retention untouched:
        # tail=[104,105,65,66,67,68] -> held = tail[:-1] = [104,105,65,66,67]
        e = self.fixture()
        get, calls = self._stepped_get(["BT 1 65", "BT 1 66", "BT 1 67", "BT 1 68", "BDONE 1 4 stop 1.0"])
        self.q.get = get
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105], born=0)
            until(lambda: not e.slot_busy[1], "healthy watcher confirms")
        self.assertEqual(e.slot_held[1], [104, 105, 65, 66, 67])
        self.assertEqual(len(calls), 5, "one watcher only")

    def test_replacement_after_the_shared_deadline_finishes_conservatively(self):
        # A03b r05 (review r3 risk): the deadline anchors at the ARM, not at each watcher's start - a BT/BDONE
        # pair that the REPLACEMENT watcher collects past the original 600 s still closes with held=[].
        e = self.fixture()
        steps = iter([(5, "boom"), (597, "BT 1 65"), (0, "BDONE 1 1 stop 1.0")])

        def scripted(timeout):
            adv, what = next(steps)
            self.now[0] += adv
            if what == "boom":
                raise RuntimeError("watcher bug")
            if what is None:
                raise queue.Empty
            return what

        self.q.get = scripted
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105], born=0)
            until(lambda: not e.slot_busy[1], "replacement confirms the end at virtual 602")
        self.assertEqual(e.slot_held[1], [], "late confirmation after the shared deadline: conservative []")
        self.assertEqual(e.slot_quarantine, set(), "the confirmed end still clears the lock")

    def test_two_dead_watchers_keep_the_quarantine_lock_never_the_reaper(self):
        # responsibility is bounded (two watchers per slot and generation, then nobody); the LOCK stands regardless:
        # busy stays, quarantine says who was last responsible, the #1603 reaper may not free a possibly-live slot.
        e = self.fixture()
        calls = []

        def boom(timeout):
            calls.append(1)
            raise RuntimeError("watcher bug")

        self.q.get = boom
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105], born=0)
            until(lambda: len(calls) >= 2 and 1 in e.slot_quarantine and not e.slot_releasing,
                  "bounded: watcher one and replacement two both ran and locked the slot")
            self.assertEqual(len(calls), 2, "exactly two watchers, no endless retry")
            self.assertTrue(e.slot_busy[1])
            self.assertEqual(e.slot_releasing, set(), "no watcher answers for it any more")
            self.assertEqual(e.slot_quarantine, {1}, "but the quarantine lock stands")
            e._release_slot_when_done(1, [104, 105], born=0)   # a hypothetical later exit of the same request
            self.assertEqual(len(calls), 2, "quarantine blocks any third watcher for this slot and generation")
            self.assertEqual(self.sent, ["BSTOP 1"], "and no second BSTOP")
            e.slot_claims.discard(1)                     # even fully ownerless, the marker protects it
            for _ in range(3):
                with self.q.cv:
                    self.now[0] += server.StrataEngine.ORPHAN_SLOT_S + 1
                with e.slot_cv:
                    self.assertEqual(e._reap_orphan_slots(), [], "never reaped: safety over availability")
                    self.assertIn(e.pick_slot([7, 8]), (0, 2), "the other slots stay available")
            self.assertTrue(e.slot_busy[1])

    def test_failed_arming_locks_the_slot_under_the_realistic_caller_handoff(self):
        # the real finally: a raise out of the arm skips the claims.discard that follows it. The slot then
        # stands busy + claimed + quarantined - locked, visible, no reaper handoff; resolved by generation end.
        e = self.fixture()

        def bad(line):
            raise ValueError("simulated closed pipe")

        e._send = bad
        with self.assertRaises(ValueError):
            e._release_slot_when_done(1, [104, 105], born=0)
        self.assertEqual(e.slot_releasing, set(), "no watcher answers for it")
        self.assertEqual(e.slot_quarantine, {1}, "the lock stands instead of a reaper handoff")
        self.assertEqual(e.slot_claims, {1}, "the finally's discard was skipped, as in production")
        e.slot_claims.discard(1)                        # then make the reap test isolate the marker itself
        with e.slot_cv:
            with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
                for _ in range(3):
                    self.now[0] += server.StrataEngine.ORPHAN_SLOT_S + 1
                    self.assertEqual(e._reap_orphan_slots(), [], "never reaped")
                self.assertIn(e.pick_slot([7, 8]), (0, 2), "the other slots stay available")
        self.assertTrue(e.slot_busy[1])

    # ---- P2 (Codex r02): the deadline is crossed by the POLL itself: retention ends at 600, not at the next loop top

    def test_deadline_crossed_during_a_poll_finishes_conservatively(self):
        # virtual-clock timelines, step-exact: (advance, line-or-None-for-Empty) per watcher get
        cases = {
            "before: BT 590->594, BDONE back at 594": ([(5, None)] * 118 + [(4, "BT 1 65"),
                                                                            (0, "BDONE 1 1 stop 1.0")],
                                                       [104, 105]),          # positive cache stands (599/594 < 600)
            "at:     poll 595->600":                  ([(5, None)] * 119 + [(5, "BDONE 1 1 stop 1.0")],
                                                       []),                   # retention ends AT the deadline
            "after:  poll 595->601":                  ([(5, None)] * 119 + [(6, "BDONE 1 1 stop 1.0")],
                                                       []),
        }
        for label, (steps, expect) in cases.items():
            self._boundary_case(steps, expect, label)

    def _boundary_case(self, steps, expect, label):
        e = self.fixture()
        now = self.now                                # closure: `self` inside TimedQ is the queue itself

        class TimedQ:
            def __init__(self, steps):
                self.steps, self.i = steps, 0

            def get(self, timeout):
                adv, line = self.steps[self.i]
                self.i += 1
                now[0] += adv
                if line is None:
                    raise queue.Empty
                return line

        e.slot_q[1] = TimedQ(steps)
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105], born=0)
            until(lambda: not e.slot_busy[1], f"release for {label}")
            self.assertEqual(e.slot_held[1], expect, label)
            self.assertEqual(e.slot_quarantine, set(), "transient marker only: the end WAS confirmed")

    def test_generation_change_ends_the_watcher_without_a_single_write(self):
        e = self.fixture()
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            until(lambda: self.q.gets >= 2, "watcher polling")
            old_busy = e.slot_busy                           # a restart replaces every array and the generation
            e.gen = 1                                        #  counter: the watcher must end on this, not drain
            e.slot_q = [None] * 3                            #  a dead queue forever nor write successor state
            e.slot_busy = [False] * 3
            e.slot_claims, e.slot_releasing, e.slot_quarantine = {1}, set(), set()
            e.slot_held = [[], [], []]
            e.slot_busy[1] = True                            # the successor request took this very slot
            time.sleep(0.1)                                  # stable-window probe: a watcher that outlives its
            g1 = self.q.gets                                 #  generation keeps polling (parent fails here), a
            time.sleep(0.3)                                  #  gated one stops within one bounded get
            self.assertEqual(self.q.gets, g1, "watcher ended at the generation's end (no endless drain)")
            self.assertEqual(e.slot_held[1], [], "successor held untouched")
            self.assertEqual(e.slot_quarantine, set(), "successor quarantine set untouched")
            self.assertTrue(e.slot_busy[1] and 1 in e.slot_claims, "successor's own claim intact")

    def test_restart_with_fewer_slots_leaves_the_old_watcher_harmless(self):
        e = self.fixture(batch=3, fast_until=610.0)
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e.slot_q[2] = self.q
            e.slot_busy[2], e.slot_claims = True, {2}
            e._release_slot_when_done(2, [104, 105])
            until(lambda: 2 in e.slot_quarantine, "quarantine")
            e.gen = 1                                        # successor comes up with only 2 slots
            e.slot_q = [FakeQ(self.now, lines=["BDONE 0 0 nonfinite 1.0"]), None]
            e.slot_busy, e.slot_claims, e.slot_releasing, e.slot_quarantine = [False, False], set(), set(), set()
            e.slot_held = [[], []]
            e.slot_order = [0, 1]                            # __init__ builds these per new batch: the successor's
            time.sleep(0.1)
            g1 = self.q.gets                                 # stable-window probe, same as the test above
            time.sleep(0.3)                                  # the old watcher ends on the generation change...
            self.assertEqual(self.q.gets, g1, "... without draining a dead queue any further")
            with e.slot_cv:
                self.assertEqual(e.pick_slot([7, 8]), 0, "the smaller successor works normally")
            self.assertEqual(len(e.slot_held), 2, "no old watcher write reached the successor arrays")

    # ---- one responsible cleanup per slot and generation ----

    def test_second_consumer_close_does_not_double_bstop_or_double_drain(self):
        e = self.fixture()
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            e._release_slot_when_done(1, [104, 105])         # the same generation's stale second exit
            self.assertEqual(self.sent, ["BSTOP 1"], "exactly one BSTOP per slot and generation")
            self.q.put("BDONE 1 1 stop 1.0")
            until(lambda: not e.slot_busy[1], "release")
            e.slot_busy[1], e.slot_claims = True, {1}         # a NEW request legitimately takes the slot later...
            q2 = e.slot_q[1] = FakeQ(self.now, lines=["BDONE 1 1 stop 1.0"])   # binding happens at entry: the
            e._release_slot_when_done(1, [104, 105])          # ... and its cleanup must arm fully on THIS queue
            self.assertEqual(self.sent, ["BSTOP 1", "BSTOP 1"], "guard must not lock the slot out forever")
            until(lambda: not q2.lines, "new watcher draining its own queue")
            until(lambda: 1 not in e.slot_releasing, "second release done")
            self.assertFalse(e.slot_busy[1], "the slot is genuinely reusable after the second confirmed end")

    # ---- all slots quarantined: nothing pickable, waiting requests still end via their own cancel ----

    def test_all_slots_quarantined_gives_no_pick(self):
        e = self.fixture(batch=2)
        e.slot_busy, e.slot_releasing, e.slot_quarantine = [True, True], {0, 1}, {0, 1}
        with e.slot_cv:
            self.assertIsNone(e.pick_slot([7, 8]))

    # ---- the reaper keeps its ordinary job around a quarantined slot ----

    def test_a_true_orphan_next_to_a_quarantined_slot_is_still_reaped(self):
        e = self.fixture(batch=3, fast_until=610.0)
        e.slot_busy[2], e.slot_claims = True, {1, 2}          # 2 leaks its flag (no owner will ever free it)
        with mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            e._release_slot_when_done(1, [104, 105])
            until(lambda: 1 in e.slot_quarantine, "quarantine")
            e.slot_claims.discard(2)                          # now 2 is the ownerless one
            with self.q.cv:
                self.now[0] += server.StrataEngine.ORPHAN_SLOT_S + 1
            with e.slot_cv:
                e._reap_orphan_slots()                        # first pass marks it seen
                with self.q.cv:
                    self.now[0] += server.StrataEngine.ORPHAN_SLOT_S + 1
                self.assertEqual(e._reap_orphan_slots(), [2], "the real orphan still gets reaped")
            self.assertTrue(e.slot_busy[1], "the quarantined one stays held")
            self.q.put(None)                                 # leave no watcher thread behind this test either
            until(lambda: not e.slot_busy[1], "watcher wind-down via EOF")

    def test_the_reaper_skips_the_quarantine_marker_on_its_own(self):
        # pins the reaper's quarantine term directly (it is belt-and-braces in production, where quarantine
        # always rides together with releasing): with only the marker set, the slot must never be reaped.
        e = self.fixture(batch=3)
        e.slot_busy = [False, True, True]
        e.slot_claims, e.slot_releasing, e.slot_quarantine = set(), set(), {2}
        with e.slot_cv, mock.patch.object(server.time, "monotonic", lambda: self.now[0]):
            self.now[0] += server.StrataEngine.ORPHAN_SLOT_S + 1
            self.assertEqual(e._reap_orphan_slots(), [], "first pass marks both seen")
            self.now[0] += server.StrataEngine.ORPHAN_SLOT_S + 1
            self.assertEqual(e._reap_orphan_slots(), [1], "1 is a real orphan; 2 carries only the marker")
        self.assertTrue(e.slot_busy[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
