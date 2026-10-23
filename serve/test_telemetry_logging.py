"""serve/test_telemetry_logging.py - the hardware sampler (serve/telemetry.py) and the engine-log readers (serve/server.py) with fakes (A12c).

No hardware, no /sys, /proc, NVML or psutil query, no real wait: the GPU readers, psutil and the CPU/RAM fallback are fakes injected before the
Telemetry object exists; its thread is a recorder that is started only by the tests that want a real one, and every wait has a deadline.  The log
readers read temp files the test wrote.  Service.start_telemetry is never called with the real sampler.

Declared contract (serve/telemetry.py's own words): "Anything that cannot be read is None; nothing here can stop the server."  So
  - a reader that raises, in any way, is "no reading" (the gpu_* keys are absent / None for that card or sample), the other series go on, and the
    sampler thread keeps sampling;
  - a disk rate is bytes per second between two counter readings: a counter that went backwards (reset, wrap) is no reading, never a negative rate;
  - the engine-log readers read each line once: two concurrent polls of one ConvCacheLog count a logged event once, and its read position never
    lies beyond the file.

Tests named ..._POLICY pin behaviour that is decided elsewhere or not at all (no staleness check, strict-JSON NaN, log rotation ...); they exist
so a change is noticed, not because it is endorsed."""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import threading
import time as real_time
import types
import unittest
from pathlib import Path
from unittest import mock

from serve import server, telemetry as T
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

DEADLINE = 10.0
REAL_THREAD = threading.Thread
ORIG_START_TELEMETRY = Service.start_telemetry         # captured at import: a foreign suite may replace it for good later


# ------------------------------------------------------------------------------------------------ fakes
class FakeGpu:
    """reads scripted answers: a dict is a reading, an exception instance is raised; the last one repeats"""
    note = None

    def __init__(self, *answers, ok=True, on_read=None):
        self.answers, self._ok, self.on_read, self.reads = list(answers), ok, on_read, 0

    def ok(self):
        return self._ok

    def name(self):
        return "fake"

    def read(self):
        if self.on_read:
            self.on_read(self)
        a = self.answers[min(self.reads, len(self.answers) - 1)]
        self.reads += 1
        if isinstance(a, BaseException):
            raise a
        return a


class Ps:
    """a psutil with scripted answers; disk is a list of (read_bytes, write_bytes) | None | exception, one per call"""

    def __init__(self, disk=(), cpu=7.0, vm=(1000, 400), cpu_exc=None):
        self.disk, self.cpu, self.vm, self.cpu_exc = list(disk), cpu, vm, cpu_exc
        self.i = 0

    def cpu_count(self, logical=True):
        return 4

    def cpu_percent(self, interval=None):
        if self.cpu_exc:
            raise self.cpu_exc
        return self.cpu

    def virtual_memory(self):
        return types.SimpleNamespace(total=self.vm[0], available=self.vm[1])

    def disk_io_counters(self):
        a = self.disk[min(self.i, len(self.disk) - 1)] if self.disk else None
        self.i += 1
        if isinstance(a, BaseException):
            raise a
        return None if a is None else types.SimpleNamespace(read_bytes=a[0], write_bytes=a[1])


class Fallback:
    def __init__(self, cpu=3.0, ram=(10, 20)):
        self._cpu, self._ram = cpu, ram

    def cpu(self):
        return self._cpu

    def ram(self):
        return self._ram


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def time(self):
        return self.t

    perf_counter = monotonic = time

    def sleep(self, s):
        self.t += s

    def __getattr__(self, n):
        return getattr(real_time, n)


class Recorder:
    """stands in for threading.Thread inside serve.telemetry: remembers the thread, runs it only on request"""
    made: list = []

    def __init__(self, target=None, daemon=None):
        self.target, self.daemon, self.real = target, daemon, None
        Recorder.made.append(self)

    def start(self):
        pass

    def run_real(self):
        self.real = REAL_THREAD(target=self.target, daemon=True)
        self.real.start()
        return self.real


class Driver:
    """a stand-in for the sampler's stop Event: the loop runs n times; each wait() is one second that did not pass"""

    def __init__(self, n, between=None):
        self.n, self.between, self.waits = n, between, []

    def is_set(self):
        return self.n <= 0

    def wait(self, t=None):
        self.waits.append(t)
        self.n -= 1
        if self.between:
            self.between()

    def set(self):
        self.n = 0


@contextlib.contextmanager
def no_hardware(psutil_mod=None, fallback=None, clock=None):
    """every input of Telemetry.__init__ is a fake; the object's thread is not started"""
    sysmods = {"psutil": psutil_mod}                    # None in sys.modules: `import psutil` raises ImportError
    proxy = types.SimpleNamespace(Thread=Recorder, Lock=threading.Lock, Event=threading.Event)
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.dict(sys.modules, sysmods))
        st.enter_context(mock.patch.object(T, "threading", proxy))
        st.enter_context(mock.patch.object(T, "_cpu_name", lambda: "cpu"))
        st.enter_context(mock.patch.object(T, "_CpuRamFallback", lambda: fallback or Fallback()))
        if clock is not None:
            st.enter_context(mock.patch.object(T, "time", clock))
        yield


def make(*gpus, ps=None, extra=None, fallback=None, clock=None, indices=None):
    readers = list(gpus)
    with no_hardware(ps, fallback, clock), mock.patch.object(T, "gpu_reader", lambda i, amd=False: readers[i]):
        t = T.Telemetry(extra=extra, gpu_index=0, gpu_indices=indices or ([0] if len(readers) < 2 else list(range(len(readers)))))
    t.thread = Recorder.made[-1]
    return t


def run_loop(t, n, between=None):
    t._stop = Driver(n, between)
    t._loop()
    return t


# ------------------------------------------------------------------------------------------------ the sampler: failures
class SamplerFailures(unittest.TestCase):
    def test_reader_exception_keeps_sampling(self):
        """Soll: a reader that raises (not an OSError: a TypeError from a half-understood file, an AttributeError ...) is 'no reading': the loop goes
        on for every second, the failed sample has no gpu_* (so nothing stale is shown), cpu/ram are still sampled, the next good read is shown."""
        g = FakeGpu({"util": 50, "temp": 60}, TypeError("unsupported operand"), {"util": 5})
        t = run_loop(make(g, ps=Ps()), 3)
        h = t.snapshot()["history"]
        self.assertEqual(g.reads, 3)                                              # the loop did not die at the second second
        self.assertEqual(h["gpu_util"], [50, None, 5])
        self.assertEqual(h["cpu"], [7.0, 7.0, 7.0])
        self.assertEqual(t.snapshot()["now"]["gpu_util"], 5)

    def test_failed_sample_leaves_no_stale_gpu_reading(self):
        g = FakeGpu({"util": 95, "mem_used": 1}, AttributeError("'NoneType' object has no attribute 'get'"))
        t = run_loop(make(g, ps=Ps()), 2)
        now = t.snapshot()["now"]
        self.assertNotIn("gpu_util", now)
        self.assertNotIn("gpu_mem_used", now)
        self.assertEqual(now["cpu"], 7.0)

    def test_multicard_one_reader_fails(self):
        """Soll: the card that raised reads as 'nothing' (the same as a card whose calls all returned None); the aggregate is over the cards that
        answered; its row in `gpus` is all None; the answering card's row is complete."""
        a = FakeGpu({"util": 40, "mem_used": 100, "mem_total": 200, "temp": 50, "power": 10.0})
        b = FakeGpu(RuntimeError("card lost"))
        t = make(a, b, ps=Ps())
        s = t.sample()
        self.assertEqual(s["gpu_util"], 40)
        self.assertEqual(s["gpu_mem_used"], 100)
        self.assertEqual(s["gpu_temp"], 50)
        rows = {r["index"]: r for r in s["gpus"]}
        self.assertEqual(rows[0], {"index": 0, "util": 40, "mem_used": 100, "mem_total": 200, "temp": 50, "power": 10.0})
        self.assertEqual(rows[1], {"index": 1, "util": None, "mem_used": None, "mem_total": None, "temp": None, "power": None})

    def test_first_card_fails_second_answers(self):
        a = FakeGpu(OSError("gone"))
        b = FakeGpu({"util": 20, "mem_used": 7, "temp": 44, "power": 5.0})
        s = make(a, b, ps=Ps()).sample()
        self.assertEqual(s["gpus"][0]["util"], None)
        self.assertEqual(s["gpus"][1]["util"], 20)

    def test_loop_survives_any_failure_of_sample(self):
        """Soll: whatever raises inside one sample, the thread lives: the sample is lost (history gets None for it), the next one is taken."""
        class Odd(Exception):                           # psutil.Error is such an Exception (not an OSError)
            pass
        ps = Ps()
        t = make(FakeGpu({"util": 1}), ps=ps)
        calls = []

        def between():
            calls.append(1)
            ps.cpu_exc = Odd("x") if len(calls) == 1 else None
        run_loop(t, 3, between)
        h = t.snapshot()["history"]
        self.assertEqual(len(h["cpu"]), 3)
        self.assertEqual(h["cpu"][0], 7.0)
        self.assertIsNone(h["cpu"][1])
        self.assertEqual(h["cpu"][2], 7.0)
        self.assertEqual(t.snapshot()["now"]["cpu"], 7.0)

    def test_extra_error_is_no_reading(self):
        def extra():
            raise ZeroDivisionError
        s = make(FakeGpu({"util": 1}), ps=Ps(), extra=extra).sample()
        self.assertEqual(s["gpu_util"], 1)
        self.assertNotIn("tok_s", s)

    def test_extra_none_and_extra_extra_keys(self):
        t = run_loop(make(FakeGpu({"util": 1}), ps=Ps(), extra=lambda: {"tok_s": None, "prefill_tok_s_mean": 12.345678, "x": 1}), 1)
        h, now = t.snapshot()["history"], t.snapshot()["now"]
        self.assertEqual(h["tok_s"], [None])
        self.assertEqual(h["prefill_tok_s_mean"], [12.35])                         # floats are rounded to 2 places in the history only
        self.assertEqual(now["prefill_tok_s_mean"], 12.345678)
        self.assertEqual(now["x"], 1)                                              # `now` carries every key `extra` returned
        self.assertNotIn("x", h)                                                  # the history only the fixed series


class NoBackend(unittest.TestCase):
    def test_no_gpu_no_psutil(self):
        t = run_loop(make(FakeGpu(ok=False), ps=None, fallback=Fallback(cpu=None, ram=(None, None))), 2)
        snap = t.snapshot()
        self.assertEqual(snap["static"]["gpu_name"], None)
        self.assertEqual(snap["static"]["gpu_count"], 1)
        self.assertEqual(snap["static"]["psutil"], False)
        self.assertEqual(snap["now"], {"cpu": None, "ram_used": None, "ram_total": None, "disk_read_mb": None, "disk_write_mb": None})
        self.assertTrue(all(v == [None, None] for v in snap["history"].values()))   # every fixed series exists, filled with None
        self.assertEqual(sorted(snap["history"]), sorted(["gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                                                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"]))

    def test_fallback_when_psutil_missing(self):
        t = make(FakeGpu(ok=False), ps=None, fallback=Fallback(cpu=12.5, ram=(5, 9)))
        s = t.sample()
        self.assertEqual((s["cpu"], s["ram_used"], s["ram_total"]), (12.5, 5, 9))
        self.assertEqual((s["disk_read_mb"], s["disk_write_mb"]), (None, None))

    def test_psutil_oserror_is_no_reading(self):
        s = make(FakeGpu(ok=False), ps=Ps(cpu_exc=OSError("denied"))).sample()
        self.assertNotIn("cpu", s)                                                 # absent, not None: the psutil branch skips the keys
        self.assertEqual(s["disk_read_mb"], None)

    def test_gpu_not_ok_reader_not_read(self):
        g = FakeGpu({"util": 99}, ok=False)
        s = make(g, ps=Ps()).sample()
        self.assertEqual(g.reads, 0)
        self.assertFalse([k for k in s if k.startswith("gpu_")])


class NonFinite(unittest.TestCase):
    def test_none_zero_missing_fields(self):
        t = run_loop(make(FakeGpu({"util": None, "mem_used": 0, "temp": 0}, {}), ps=Ps()), 2)
        h = t.snapshot()["history"]
        self.assertEqual(h["gpu_util"], [None, None])
        self.assertEqual(h["gpu_mem_used"], [0, None])
        self.assertEqual(h["gpu_temp"], [0, None])

    def test_nan_inf_pass_through_POLICY(self):
        """POLICY: a non-finite reading is stored as it came (no filter anywhere in the sampler); json.dumps writes NaN / Infinity, which is not JSON
        a strict parser (JSON.parse in a browser) accepts.  Real readers produce int / finite float; only a foreign file (the Xe reader's gpustat
        JSON) can carry one.  A filter would be a new data policy: proposal P3."""
        t = run_loop(make(FakeGpu({"util": float("nan"), "temp": float("inf")}), ps=Ps()), 1)
        now = t.snapshot()["now"]
        self.assertTrue(math.isnan(now["gpu_util"]) and math.isinf(now["gpu_temp"]))
        self.assertIn("NaN", json.dumps(now))
        with self.assertRaises(ValueError):
            json.dumps(now, allow_nan=False)

    def test_multicard_aggregates_skip_none(self):
        a = FakeGpu({"util": 10, "mem_used": 1, "temp": 30, "power": None})
        b = FakeGpu({"util": None, "mem_used": 2, "temp": 70, "power": 4.0})
        s = make(a, b, ps=Ps()).sample()
        self.assertEqual((s["gpu_util"], s["gpu_mem_used"], s["gpu_temp"], s["gpu_power"]), (10, 3, 70, 4.0))

    def test_multicard_all_none_is_none(self):
        s = make(FakeGpu({}), FakeGpu({}), ps=Ps()).sample()
        self.assertEqual((s["gpu_util"], s["gpu_mem_used"], s["gpu_temp"], s["gpu_power"]), (None, None, None, None))

    def test_gpu_count_and_not_ok_card_dropped(self):
        t = make(FakeGpu({"util": 1}), FakeGpu({"util": 2}, ok=False), ps=Ps())
        self.assertEqual(t.static["gpu_count"], 1)                                  # a card that is not ok is left out of the list


# ------------------------------------------------------------------------------------------------ the sampler: disk rates
class DiskRates(unittest.TestCase):
    MB = 2 ** 20

    def sampler(self, series):
        clock = Clock()
        t = make(FakeGpu(ok=False), ps=Ps(disk=series), clock=clock)
        patch = mock.patch.object(T, "time", clock)             # the clock stays the fake one for the whole test
        patch.start()
        self.addCleanup(patch.stop)
        return t, clock

    def test_rates(self):
        t, c = self.sampler([(0, 0), (10 * self.MB, 2 * self.MB), (30 * self.MB, 2 * self.MB)])
        self.assertEqual(t._disk(), (None, None))                                  # the first reading has nothing to compare with
        c.t += 1
        self.assertEqual(t._disk(), (10.0, 2.0))
        c.t += 2
        self.assertEqual(t._disk(), (10.0, 0.0))

    def test_counter_reset_is_no_reading(self):
        """Soll: a counter that went backwards (a reset, a wrap, a changed device set) is no reading for that interval - never a negative rate -
        and the next interval is measured from the new value."""
        t, c = self.sampler([(100 * self.MB, 50 * self.MB), (110 * self.MB, 50 * self.MB), (5 * self.MB, 1 * self.MB), (15 * self.MB, 1 * self.MB)])
        t._disk()
        c.t += 1
        self.assertEqual(t._disk(), (10.0, 0.0))
        c.t += 1
        r, w = t._disk()
        self.assertTrue(r is None and w is None, (r, w))
        c.t += 1
        self.assertEqual(t._disk(), (10.0, 0.0))

    def test_one_counter_reset_other_fine(self):
        t, c = self.sampler([(100 * self.MB, 10 * self.MB), (5 * self.MB, 20 * self.MB)])
        t._disk()
        c.t += 1
        r, w = t._disk()
        self.assertIsNone(r)
        self.assertEqual(w, 10.0)

    def test_clock_backward_and_equal(self):
        t, c = self.sampler([(0, 0), (self.MB, 0), (2 * self.MB, 0), (3 * self.MB, 0)])
        t._disk()
        c.t += 1
        self.assertEqual(t._disk()[0], 1.0)
        c.t -= 5                                                                   # the wall clock jumped back
        self.assertEqual(t._disk(), (None, None))
        self.assertEqual(t._disk(), (None, None))                                  # same time again: dt = 0, no division
        c.t += 1
        r = t._disk()[0]
        self.assertTrue(r is not None and r >= 0 and math.isfinite(r), r)

    def test_disk_errors_and_none(self):
        t, c = self.sampler([OSError("x"), RuntimeError("y"), None, (self.MB, 0)])
        for _ in range(4):
            c.t += 1
            r = t._disk()
        self.assertEqual(t._disk_prev[1:], (self.MB, 0))                           # the errors left no half state: the next reading is the baseline


# ------------------------------------------------------------------------------------------------ snapshot, history, threads
class SnapshotAndHistory(unittest.TestCase):
    def test_history_bounded_and_ordered(self):
        n = [0]
        t = make(FakeGpu(ok=False), ps=Ps(), extra=lambda: {"tok_s": n.__setitem__(0, n[0] + 1) or n[0]})
        run_loop(t, T.HISTORY + 7)
        h = t.snapshot()["history"]["tok_s"]
        self.assertEqual(len(h), T.HISTORY)
        self.assertEqual(h, list(range(8, T.HISTORY + 8)))                         # the oldest 7 are gone, order kept

    def test_snapshot_is_isolated_from_the_sampler(self):
        t = run_loop(make(FakeGpu({"util": 1}), ps=Ps()), 2)
        a = t.snapshot()
        a["now"]["gpu_util"] = 999
        a["history"]["cpu"].append("x")
        a["static"]["gpu_name"] = "changed"
        b = t.snapshot()
        self.assertEqual(b["now"]["gpu_util"], 1)
        self.assertEqual(b["history"]["cpu"], [7.0, 7.0])
        self.assertEqual(b["static"]["gpu_name"], "fake")
        self.assertIsNot(a["history"]["cpu"], b["history"]["cpu"])

    def test_snapshot_nested_gpus_list_is_shared_POLICY(self):
        """POLICY: snapshot() copies one level; with more than one card `now["gpus"]` (a list of dicts) is the sampler's own object.  No caller
        mutates it (metrics() hands it to json.dumps); a deep copy would cost a copy per request for nobody."""
        t = run_loop(make(FakeGpu({"util": 1}), FakeGpu({"util": 2}), ps=Ps()), 1)
        a, b = t.snapshot(), t.snapshot()
        self.assertIs(a["now"]["gpus"], b["now"]["gpus"])
        self.assertIs(a["now"]["gpus"], t.now["gpus"])

    def test_snapshot_before_first_sample(self):
        t = make(FakeGpu({"util": 1}), ps=Ps())
        self.assertEqual(t.snapshot()["now"], {})
        self.assertEqual(t.snapshot()["history"], {})


class RealThread(unittest.TestCase):
    """The loop on a real thread, gated by events the fake reader sets; every wait has a deadline."""

    def start(self, gpu, **kw):
        t = make(gpu, ps=Ps(), **kw)
        th = t.thread.run_real()
        self.addCleanup(self.finish, t, th)
        return t, th

    @staticmethod
    def finish(t, th):
        t.close()
        th.join(DEADLINE)

    def test_snapshot_does_not_wait_for_a_sample(self):
        """the lock is not held while a reader reads: snapshot() answers (the previous reading) while a read is in progress"""
        inside, release = threading.Event(), threading.Event()

        def on_read(g):
            if g.reads == 1:                                    # the second sample blocks
                inside.set()
                release.wait(DEADLINE)
        g = FakeGpu({"util": 33}, on_read=on_read)
        t, th = self.start(g)
        self.assertTrue(inside.wait(DEADLINE))                  # sample 1 done, sample 2 inside read()
        # wait for sample 1 to be visible (it is: the loop appended it before waiting 1 s for sample 2)
        done = []
        q = threading.Thread(target=lambda: done.append(t.snapshot()), daemon=True)
        q.start()
        q.join(DEADLINE)
        release.set()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["now"]["gpu_util"], 33)

    def test_close_while_waiting_ends_the_thread(self):
        first = threading.Event()
        g = FakeGpu({"util": 1}, on_read=lambda g: first.set())
        t, th = self.start(g)
        self.assertTrue(first.wait(DEADLINE))
        t.close()
        th.join(DEADLINE)
        self.assertFalse(th.is_alive())
        n = g.reads
        self.assertEqual(t.snapshot()["now"]["gpu_util"], 1)  # still readable after close
        self.assertEqual(g.reads, n)

    def test_close_during_a_blocking_read_cannot_end_it(self):
        """The limit: close() only sets an event.  A reader blocked in a foreign call (NVML, a hung file system) keeps the thread inside it - Python
        cannot end a thread, and no timeout or join policy is invented here.  Once the call returns, the thread takes no further sample."""
        inside, release = threading.Event(), threading.Event()

        def on_read(g):
            inside.set()
            release.wait(DEADLINE)
        g = FakeGpu({"util": 8}, on_read=on_read)
        t, th = self.start(g)
        self.assertTrue(inside.wait(DEADLINE))
        t.close()                                                 # returns at once ...
        th.join(0.3)
        self.assertTrue(th.is_alive())                            # ... the thread is still inside the reader
        release.set()
        th.join(DEADLINE)
        self.assertFalse(th.is_alive())
        self.assertEqual(g.reads, 1)                              # and it did not read again
        self.assertEqual(t.snapshot()["now"]["gpu_util"], 8)      # the sample in flight was still recorded

    def test_close_twice_and_before_start(self):
        t = make(FakeGpu({"util": 1}), ps=Ps())
        t.close()
        t.close()
        th = t.thread.run_real()
        th.join(DEADLINE)
        self.assertFalse(th.is_alive())
        self.assertEqual(t.snapshot()["history"], {})              # a closed sampler takes no sample at all

    def test_no_thread_left(self):
        g = FakeGpu({"util": 1})
        t, th = self.start(g)
        t.close()
        th.join(DEADLINE)
        self.assertNotIn(th, threading.enumerate())

    def test_parallel_snapshots_while_sampling(self):
        stop = threading.Event()
        seen, errors = [], []

        def reader():
            while not stop.is_set():
                try:
                    s = t.snapshot()
                    seen.append(len(s["history"].get("cpu", [])))
                    lens = {len(v) for v in s["history"].values()}
                    assert len(lens) <= 1, lens                    # one sample appends to every series under one lock
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                    return
        ticks = threading.Event()
        counter = [0]

        def on_read(g):
            counter[0] += 1
            if counter[0] >= 3:
                ticks.set()
        g = FakeGpu({"util": 1}, on_read=on_read)
        t = make(g, ps=Ps())
        t._stop = Driver(5)                                        # no real second between samples
        th = REAL_THREAD(target=t._loop, daemon=True)
        q = REAL_THREAD(target=reader, daemon=True)
        q.start()
        th.start()
        th.join(DEADLINE)
        stop.set()
        q.join(DEADLINE)
        self.assertFalse(errors, errors)
        self.assertEqual(len(t.snapshot()["history"]["cpu"]), 5)
        self.assertTrue(seen)


# ------------------------------------------------------------------------------------------------ the server's side of it
class FakeTelemetry:
    made: list = []

    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False):
        self.extra, self.closed, self.now = extra, 0, {}
        FakeTelemetry.made.append(self)

    def snapshot(self):
        return {"now": dict(self.now), "history": {}, "static": {}}

    def close(self):
        self.closed += 1


class GpuEngine(MockEngine):
    gpu_busy = None
    _stall_check = server.StrataEngine._stall_check


def new_service():
    tok = ByteTokenizer()
    return Service(GpuEngine(tok, "x", max_context=4096), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))


def real_start_telemetry():
    if getattr(ORIG_START_TELEMETRY, "__name__", "") == "<lambda>":
        raise unittest.SkipTest("Service.start_telemetry was replaced for good by a foreign suite before this module was imported")
    return ORIG_START_TELEMETRY


class GpuBusyDecision(unittest.TestCase):
    """engine.gpu_busy (installed by Service.start_telemetry) decides, in the engine's frozen check, whether silence with no CPU / IO is a freeze"""

    def setUp(self):
        FakeTelemetry.made.clear()
        self.svc = new_service()
        with mock.patch.object(T, "Telemetry", FakeTelemetry):
            real_start_telemetry()(self.svc)
        self.tel = FakeTelemetry.made[-1]
        self.clock = Clock()
        self.trace = []

    def decide(self, now_dict, activity=((1.0, 100), (1.0, 100))):
        """one frozen check at 100 s of silence with the engine's CPU/IO unchanged; -> 'ended' | 'kept'"""
        eng = self.svc.engine
        self.tel.now = now_dict
        seq = iter(activity)
        eng._activity = lambda: next(seq)
        eng._silent = lambda what: server.EngineSilent(what)
        state = {}
        out = io.StringIO()
        with mock.patch.object(server, "time", self.clock), contextlib.redirect_stdout(out):
            eng._stall_check(self.clock.t - 100, state, 60.0)     # first call: the baseline
            try:
                eng._stall_check(self.clock.t - 100, state, 60.0)
            except server.EngineSilent:
                return "ended", out.getvalue()
        return "kept", out.getvalue()

    def test_wiring(self):
        self.assertTrue(callable(self.svc.engine.gpu_busy))
        real_start_telemetry()(self.svc)                           # again: the sampler is made once
        self.assertEqual(len(FakeTelemetry.made), 1)
        self.assertIsNotNone(self.svc.telemetry.extra)

    def test_matrix(self):
        cases = [({"gpu_util": 10}, "kept"), ({"gpu_util": 100}, "kept"), ({"gpu_util": 9.9}, "ended"), ({"gpu_util": 0}, "ended"),
                 ({"gpu_util": None}, "ended"), ({}, "ended"), ({"gpu_util": float("nan")}, "ended"), ({"gpu_util": float("inf")}, "kept"),
                 ({"gpu_util": "50"}, "ended")]
        for now, want in cases:
            with self.subTest(now=now):
                got, said = self.decide(now)
                self.assertEqual(got, want)
                if want == "kept":
                    self.assertIn("still working", said)

    def test_activity_still_decides_when_gpu_idle(self):
        got, _ = self.decide({"gpu_util": 0}, activity=((1.0, 100), (2.0, 100)))
        self.assertEqual(got, "kept")                              # CPU moved: working

    def test_snapshot_error_is_a_hiccup(self):
        eng = self.svc.engine
        self.tel.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("closed"))
        seq = iter(((1.0, 1), (1.0, 1)))
        eng._activity = lambda: next(seq)
        eng._silent = lambda what: server.EngineSilent(what)
        state = {}
        with mock.patch.object(server, "time", self.clock), contextlib.redirect_stdout(io.StringIO()):
            eng._stall_check(self.clock.t - 100, state, 60.0)
            with self.assertRaises(server.EngineSilent):
                eng._stall_check(self.clock.t - 100, state, 60.0)

    def test_stale_reading_keeps_the_engine_POLICY(self):
        """POLICY: gpu_busy reads the sampler's last `now` with no age check.  A sampler that stopped (closed, or blocked in a reader) leaves the last
        gpu_util in place, so a last reading >= 10 % stops the frozen-engine decision for as long as nothing replaces it.  A staleness limit is a
        new policy (proposal P2); with the reader-failure fix a *failing* reader no longer leaves one behind."""
        self.tel.now = {"gpu_util": 90}
        for _ in range(3):
            self.assertTrue(self.svc.engine.gpu_busy())


class ServerCloseStopsSampler(unittest.TestCase):
    def test_server_close_closes_the_sampler_once(self):
        FakeTelemetry.made.clear()
        with mock.patch.object(T, "Telemetry", FakeTelemetry), \
                mock.patch.object(Service, "start_telemetry", real_start_telemetry()):
            svc = new_service()
            httpd = serve(svc, port=0)
        try:
            self.assertEqual(len(FakeTelemetry.made), 1)
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(FakeTelemetry.made[0].closed, 1)

    def test_server_close_without_sampler(self):
        with mock.patch.object(Service, "start_telemetry", lambda self: None):
            httpd = serve(new_service(), port=0)
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------------------------------------------------------------ the log readers
EVENT = b"strata serve: conversation cache: parked 100 tokens in 5 ms; parked=1 bytes=1000 evictions=0\n"
RESTORE = b"strata serve: conversation cache: restored 80 tokens in 2 ms; parked=0 bytes=0\n"


def state_of(**kw):
    s = {"parked": 0, "bytes": 0, "evictions": 0, "parks": 0, "restores": 0, "last_event": None, "last_tokens": None, "last_at": None}
    s.update(kw)
    return s


class TempLog(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "engine.log")
        self.clock = Clock()

    def write(self, data, mode="ab"):
        with open(self.path, mode) as f:
            f.write(data)


class ConvCacheLogTests(TempLog):
    def poll(self, c, start=0):
        with mock.patch.object(server, "time", self.clock):
            return c.poll(self.path, start)

    def test_no_log_and_no_start(self):
        c = server.ConvCacheLog()
        self.assertEqual(c.poll(None, 0), state_of())
        self.assertEqual(c.poll(self.path, None), state_of())
        self.assertEqual(self.poll(c), state_of())                                  # the file does not exist: the empty state, no raise

    def test_events_once_and_follow_up(self):
        self.write(b"[strata] x\n" + EVENT)
        c = server.ConvCacheLog()
        want = state_of(parked=1, bytes=1000, evictions=0, parks=1, last_event="parked", last_tokens=100, last_at=self.clock.t)
        self.assertEqual(self.poll(c), want)
        self.assertEqual(self.poll(c), want)                                        # polled again: nothing new, nothing counted twice
        self.write(RESTORE)
        self.assertEqual(self.poll(c), state_of(parked=0, bytes=0, evictions=0, parks=1, restores=1, last_event="restored", last_tokens=80,
                                                last_at=self.clock.t))

    def test_partial_line_waits_for_its_newline(self):
        c = server.ConvCacheLog()
        self.write(EVENT[:40])
        self.assertEqual(self.poll(c), state_of())
        self.write(EVENT[40:])
        self.assertEqual(self.poll(c)["parks"], 1)                                  # the half line was kept for the next read: neither lost nor twice
        self.assertEqual(self.poll(c)["parks"], 1)

    def test_utf8_boundary_between_polls(self):
        c = server.ConvCacheLog()
        line = "note ä€😀 conversation cache: parked 5 tokens; parked=2 bytes=30\n".encode()
        cut = line.index("€".encode()) + 1                                         # inside the 3-byte character
        self.write(line[:cut])
        self.assertEqual(self.poll(c)["parks"], 0)
        self.write(line[cut:])
        st = self.poll(c)
        self.assertEqual((st["parks"], st["parked"], st["bytes"], st["evictions"]), (1, 2, 30, 0))

    def test_invalid_utf8_line_does_not_stop_the_reader(self):
        c = server.ConvCacheLog()
        self.write(b"\xff\xfe garbage\n" + EVENT)
        self.assertEqual(self.poll(c)["parks"], 1)

    def test_skipped_and_dropped_events(self):
        c = server.ConvCacheLog()
        self.write(b"x conversation cache: skipped 10 tokens; parked=3 bytes=7 evictions=2\n"
                   b"x conversation cache: dropped 2 superseded entries; parked=9\n")
        st = self.poll(c)
        self.assertEqual((st["parked"], st["bytes"], st["evictions"], st["parks"], st["restores"], st["last_event"]), (9, 7, 2, 0, 0, None))

    def test_long_line_without_newline_then_complete(self):
        c = server.ConvCacheLog()
        self.write(b"a" * (3 << 20))
        self.assertEqual(self.poll(c), state_of())
        self.assertLessEqual(c.pos, os.path.getsize(self.path))
        self.write(b"\n" + EVENT)
        self.assertEqual(self.poll(c)["parks"], 1)

    def test_read_cap_skips_old_lines(self):
        """the limit: at most the last READ_MAX bytes of new lines are read; events before that are not counted"""
        class Small(server.ConvCacheLog):
            READ_MAX = len(EVENT) * 2 + 3
        c = Small()
        self.write(EVENT * 5)
        st = self.poll(c)
        self.assertEqual(st["parks"], 2)                                            # the cap cuts into the 4th line (dropped) and keeps 2 whole lines
        self.assertEqual(c.pos, os.path.getsize(self.path))

    def test_truncation_rereads_from_the_start(self):
        c = server.ConvCacheLog()
        self.write(EVENT * 3)
        self.assertEqual(self.poll(c)["parks"], 3)
        self.write(RESTORE, "wb")                                                   # emptied and written again, shorter than the old position
        st = self.poll(c)
        self.assertEqual((st["parks"], st["restores"]), (3, 1))                    # the counters keep counting (they are totals of this run)
        self.assertEqual(c.pos, os.path.getsize(self.path))

    def test_replaced_longer_file_is_not_noticed_POLICY(self):
        """POLICY: there is no inode / generation check: a log replaced by a longer one is read on from the old position (the engine only
        appends, and a new run is a new `start`)."""
        c = server.ConvCacheLog()
        self.write(EVENT)
        self.poll(c)
        self.write(b"y" * len(EVENT) + EVENT * 2, "wb")
        self.assertEqual(self.poll(c)["parks"], 3)                                  # the first new line (cut at the old position) is not read

    def test_new_start_resets(self):
        c = server.ConvCacheLog()
        self.write(EVENT)
        self.assertEqual(self.poll(c, 0)["parks"], 1)
        self.write(RESTORE)
        st = self.poll(c, os.path.getsize(self.path) - len(RESTORE))
        self.assertEqual((st["parks"], st["restores"]), (0, 1))

    def test_read_error_keeps_the_state(self):
        c = server.ConvCacheLog()
        self.write(EVENT)
        want = self.poll(c)
        os.remove(self.path)
        self.assertEqual(self.poll(c), want)
        os.mkdir(self.path)                                                         # a directory where the log was: open fails (OSError)
        self.assertEqual(self.poll(c), want)

    def test_returned_state_is_a_copy(self):
        c = server.ConvCacheLog()
        self.write(EVENT)
        a = self.poll(c)
        a["parks"] = 99
        self.assertEqual(self.poll(c)["parks"], 1)

    def test_concurrent_polls_count_an_event_once(self):
        """Soll: metrics() runs on one handler thread per request; two polls at the same moment must not both read the same bytes (events counted
        twice, the position beyond the end of the file, the next poll starting over)."""
        self.write(b"[strata] x\n" + EVENT)
        c = server.ConvCacheLog()
        barrier = threading.Barrier(2, timeout=0.5)
        real_open = open

        def fake_open(path, *a, **k):
            f = real_open(path, *a, **k)
            if path == self.path:
                inner = f.read

                def read(n=-1):
                    data = inner(n)
                    try:
                        barrier.wait()                                              # both pollers hold the same bytes before either updates
                    except threading.BrokenBarrierError:
                        pass                                                        # the other poller waits for a lock: no overlap possible
                    return data
                f.read = read
            return f
        out = []
        with mock.patch.object(server, "open", fake_open, create=True), mock.patch.object(server, "time", self.clock):
            ts = [REAL_THREAD(target=lambda: out.append(c.poll(self.path, 0)), daemon=True) for _ in range(2)]
            [t.start() for t in ts]
            [t.join(DEADLINE) for t in ts]
        self.assertEqual(len(out), 2)
        self.assertEqual(c.state["parks"], 1)
        self.assertEqual(c.pos, os.path.getsize(self.path))
        self.assertEqual(self.poll(c)["parks"], 1)


class StartLogReaders(TempLog):
    def test_tail_basic_and_offset(self):
        self.write(b"old 1\nold 2\n")
        off = os.path.getsize(self.path)
        self.write(b"a\n\n  \nb\nc  \n")
        self.assertEqual(server.start_log_tail(self.path, off), "\nthe engine log's last lines:\n  a\n  b\n  c")
        self.assertEqual(server.start_log_tail(self.path, off, n=2), "\nthe engine log's last lines:\n  b\n  c")
        self.assertEqual(server.start_log_tail(self.path, os.path.getsize(self.path)), "")

    def test_tail_no_log_empty_beyond_end_and_errors(self):
        self.assertEqual(server.start_log_tail(None, 0), "")
        self.assertEqual(server.start_log_tail(self.path, 0), "")                   # missing
        self.write(b"")
        self.assertEqual(server.start_log_tail(self.path, 0), "")                   # empty
        self.write(b"x\n")
        self.assertEqual(server.start_log_tail(self.path, 10_000), "")              # offset beyond the end (the log was cut)
        self.assertEqual(server.start_log_tail(self.dir.name, 0), "")               # a directory: OSError

    def test_tail_window_is_64k_and_never_an_earlier_start(self):
        self.write(b"earlier start\n" + b"".join(b"line %05d\n" % i for i in range(20000)))
        off = len(b"earlier start\n")
        t = server.start_log_tail(self.path, off)
        self.assertTrue(t.endswith("  line 19999"))
        self.assertEqual(len(t.splitlines()), 22)                                   # "", the header, 20 lines
        self.assertNotIn("earlier", t)

    def test_tail_utf8_cut_and_invalid(self):
        self.write("x\n".encode() + ("ä" * 40000 + "\n").encode() + b"\xff\xfe end\n")
        t = server.start_log_tail(self.path, 0)
        self.assertTrue(t.endswith("�� end"))                             # invalid bytes are replacement characters, the call returns

    def test_tail_one_long_line_is_dropped_POLICY(self):
        """POLICY/limit: the window starts mid-line when the log since the start is longer than 64 KiB, and that first (cut) line is dropped; if the
        whole window is one line (a single >64 KiB message), the tail is empty although the log has content."""
        self.write(b"before\n" + b"x" * 200_000 + b"\n")
        self.assertEqual(server.start_log_tail(self.path, 7), "")

    def test_marks_window_and_errors(self):
        self.write(b"[strata] started\nterminate called after throwing hsa_exception\n")
        self.assertEqual(server.start_log_marks(self.path, 0), ["hsa_exception"])
        self.assertEqual(server.start_log_marks(self.path, 10_000), [])
        self.assertEqual(server.start_log_marks(None, 0), [])
        self.assertEqual(server.start_log_marks(self.dir.name, 0), [])

    def test_marks_beyond_64k_are_not_seen_POLICY(self):
        self.write(b"x" * (64 * 1024 + 10) + b"\nhsa_exception\n")
        self.assertEqual(server.start_log_marks(self.path, 0), [])

    def test_transient_failure_matrix(self):
        self.write(b"[strata] engine started\n")
        self.assertTrue(server.transient_start_failure(self.path, 0, 5, None))     # nothing of its own, no exit code: a runtime start
        self.assertFalse(server.transient_start_failure(self.path, 0, 31, None))   # too late
        self.assertFalse(server.transient_start_failure(self.path, 0, 5, 1))      # the engine's own error exit
        self.write(b"what(): HSA_STATUS_ERROR\nterminate called\n")
        self.assertTrue(server.transient_start_failure(self.path, 0, 5, -6))
        self.write(b"loading weights\n")
        self.assertFalse(server.transient_start_failure(self.path, 0, 5, -6))     # the engine said something of its own as well

    def test_failure_hint(self):
        self.write(b"[strata] x\n")
        self.assertEqual(server.start_failure_hint(self.path, 0), "")
        self.write(b"the draft head does not fit\nstrata mtp: draft head over by 100 MiB\nstrata mtp: hint: use --draft-vocab en\n")
        h = server.start_failure_hint(self.path, 0)
        self.assertIn("draft head over by 100 MiB hint: use --draft-vocab en", h)
        self.assertEqual(server.start_failure_hint(self.path, 10_000), "")
        self.assertEqual(server.start_failure_hint(None, 0), "")
        self.assertEqual(server.start_failure_hint(self.dir.name, 0), "")
        self.write(b"the draft head does not fit\n", "wb")
        self.assertIn("Hint: ", server.start_failure_hint(self.path, 0))          # an older engine: the same advice in words


class Halt(Exception):
    pass


class FollowerTests(TempLog):
    def run_echo(self, steps, offset=0):
        """echo_requests on the log; steps[i] is done at the i-th sleep (appends); the loop ends when the steps are used up"""
        it = iter(steps)

        class T_:
            def __getattr__(s, n):
                return getattr(real_time, n)

            def sleep(s, t):
                try:
                    step = next(it)
                except StopIteration:
                    raise Halt
                step()
        out = io.StringIO()
        with mock.patch.object(server, "time", T_()), contextlib.redirect_stdout(out):
            with self.assertRaises(Halt):
                server.echo_requests(self.path, offset)
        return out.getvalue()

    REQ = (b"strata serve: prompt 100 tokens = 40 reused + 60 read in 120.0 ms (500.0 tok/s), 20 generated in 400.0 ms (50.0 tok/s)\n")
    WANT = ("[strata] request prompt 100 cached 40 output 20 prompt_read 120 ms total 520 ms prefill 500.0 tok/s decode 50.0 tok/s\n")

    def test_echo_each_request_once_in_order(self):
        self.write(b"noise\n" + self.REQ)
        out = self.run_echo([lambda: self.write(self.REQ.replace(b"100 tokens", b"7 tokens")), lambda: None])
        lines = out.splitlines(keepends=True)
        self.assertEqual(lines[0], self.WANT)
        self.assertTrue(lines[1].startswith("[strata] request prompt 7 cached 40"))
        self.assertEqual(len(lines), 2)

    def test_echo_starts_at_the_offset(self):
        self.write(self.REQ)
        off = os.path.getsize(self.path)
        self.assertEqual(self.run_echo([lambda: None], off), "")

    def test_echo_line_split_over_two_reads_is_lost_POLICY(self):
        """POLICY/limit: readline() returns what is there; a request line written in two pieces is matched as neither piece, so no echo line.  The
        engine writes a line in one write, so the window is the time between a write's first and last byte becoming visible - nothing here can
        close it without buffering partial lines (proposal P4)."""
        half = len(self.REQ) // 2
        self.write(self.REQ[:half])
        out = self.run_echo([lambda: self.write(self.REQ[half:]), lambda: None])
        self.assertEqual(out, "")

    def test_echo_after_truncation_is_silent_POLICY(self):
        """POLICY/limit: the follower keeps its file offset: a log emptied and rewritten shorter is not followed (the engine appends, and a start
        never empties the log: StrataEngine opens it with 'a')."""
        self.write(b"x" * 500 + self.REQ)
        off = os.path.getsize(self.path)
        out = self.run_echo([lambda: self.write(self.REQ, "wb"), lambda: None], off)
        self.assertEqual(out, "")

    def test_echo_invalid_utf8_continues(self):
        self.write(b"\xff\xfe\n" + self.REQ)
        self.assertEqual(self.run_echo([lambda: None]), self.WANT)

    def run_narrate(self, steps, args=None, heartbeat=20.0, offset=0):
        clock = Clock()
        it = iter(steps)

        class Done:
            def wait(s, t):
                try:
                    step = next(it)
                except StopIteration:
                    return True
                step(clock)
                return False
        out = io.StringIO()
        with mock.patch.object(server, "time", clock), contextlib.redirect_stdout(out):
            server.narrate_start(self.path, offset, args or [], Done(), heartbeat)
        return out.getvalue()

    def test_narrate_messages_once_and_in_order(self):
        self.write(b"")
        steps = [lambda c: self.write(b"engine PLE on\nexpert arena: 10 GiB\n"),
                 lambda c: self.write(b"experts loaded 50.0 GiB at 1.2 GiB/s\n"),
                 lambda c: self.write(b"expert cache 128 slots, 40 MiB each; x\n"),
                 lambda c: self.write(b"session is up\nsession is up\n")]
        out = self.run_narrate(steps)
        self.assertEqual(out.count("YOUR PC CAN BE SLOW"), 1)                      # PLE on and expert arena: said once
        self.assertIn("experts loaded: 50.0 GiB at 1.2 GiB/s", out)
        self.assertIn("filling the GPU's expert cache (128 experts, 40 MiB each)", out)
        self.assertEqual(out.count("almost ready"), 1)
        self.assertLess(out.index("YOUR PC"), out.index("experts loaded"))

    def test_narrate_partial_line_waits(self):
        self.write(b"")
        out = self.run_narrate([lambda c: self.write(b"session is "), lambda c: self.write(b"up\n")])
        self.assertEqual(out.count("almost ready"), 1)

    def test_narrate_read_error_and_missing_log(self):
        out = self.run_narrate([lambda c: None, lambda c: None])                   # no log file at all
        self.assertEqual(out.strip(), "[strata] starting the engine: reading the model's weights ...")

    def test_narrate_heartbeat(self):
        self.write(b"")
        out = self.run_narrate([lambda c: c.sleep(25), lambda c: c.sleep(25)], heartbeat=20.0)
        self.assertEqual(out.count("still starting"), 2)

    def test_narrate_after_truncation_is_silent_POLICY(self):
        self.write(b"x" * 1000)
        out = self.run_narrate([lambda c: self.write(b"session is up\n", "wb")], offset=1000)
        self.assertNotIn("almost ready", out)


if __name__ == "__main__":
    unittest.main()
