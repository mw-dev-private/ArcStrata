"""serve/test_service_lifecycle.py - Service.load / ensure_loaded / unload / unload_waiting / start_idle_unload against recording fakes (A11b).

Engine, vision encoder, the before_load command (subprocess.run) and the free-VRAM reading are fakes that write one shared trace; no process,
hook or hardware is touched.  Every case checks the whole state afterwards (engine, encoder, fifo, queued/busy counters, vision_held, call
trace) and that a healthy next call works.  Tests named ..._POLICY pin behaviour that is an open question; they exist so a change is noticed,
not because it is endorsed."""
import itertools
import subprocess
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineStuck, GpuBusy, MockEngine, Service

DEADLINE = 10.0


class Engine(MockEngine):
    def __init__(self, trace, state="alive", fail_restart=None, fail_unload=None):
        super().__init__(ByteTokenizer(), "x", max_context=4096)
        self.trace, self.state, self.fail_restart, self.fail_unload = trace, state, fail_restart, fail_unload
        self.starts, self.restart_gate, self.restart_entered = 0, None, threading.Event()

    @property
    def unloaded(self):
        return self.state == "unloaded"

    def alive(self):
        return self.state == "alive"

    def exit_code(self):
        return 9

    def unload(self):
        self.trace.append("engine.unload")
        if self.fail_unload:
            raise self.fail_unload
        self.state = "unloaded"

    def restart(self):
        self.trace.append("engine.restart")
        self.restart_entered.set()
        if self.restart_gate is not None:
            assert self.restart_gate.wait(DEADLINE), "restart gate never opened"
        if self.fail_restart:
            raise self.fail_restart
        self.state = "alive"
        self.starts += 1


class Vision:
    elastic = False

    def __init__(self, trace, alive=True, fail_restart=None, fail_unload=None):
        self.trace, self._alive, self.fail_restart, self.fail_unload = trace, alive, fail_restart, fail_unload

    def alive(self):
        return self._alive

    def restart(self):
        self.trace.append("vision.restart")
        if self.fail_restart:
            raise self.fail_restart
        self._alive = True

    def unload(self):
        self.trace.append("vision.unload")
        if self.fail_unload:
            raise self.fail_unload
        self._alive = False

    def shutdown(self):
        pass


class Hook:
    """stands in for subprocess.run of the before_load command"""

    def __init__(self, trace, outcome="ok"):
        self.trace, self.outcome, self.calls = trace, outcome, []

    def __call__(self, cmd, **kw):
        self.calls.append({"cmd": cmd, "shell": kw.get("shell"), "timeout": kw.get("timeout")})
        self.trace.append("hook")
        if self.outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        if self.outcome == "oserror":
            raise OSError(2, "no such file")
        return subprocess.CompletedProcess(cmd, 0 if self.outcome == "ok" else int(self.outcome))

    def patch(self):
        return mock.patch.object(server.subprocess, "run", self)


class Lifecycle(unittest.TestCase):
    def make(self, engine_state="alive", vision=None, **ekw):
        Service.start_telemetry = lambda self: None            # before anything can start
        self.trace = []
        self.eng = Engine(self.trace, engine_state, **ekw)
        self.svc = Service(self.eng, ByteTokenizer(), ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        self.vis = None
        if vision is not None:
            self.vis = Vision(self.trace, **vision)
            self.svc.vision = self.vis
        self.svc.vram_free_wait_s = 0.0
        return self.svc

    def state(self):
        v = self.vis
        return {"engine": self.eng.state, "vision": None if v is None else ("up" if v._alive else "down"), "fifo_locked": self.svc.fifo.locked(),
                "queued": self.svc.status["queued"], "busy": self.svc.status["busy"], "vision_held": self.svc.vision_held, "trace": list(self.trace)}

    def assertClean(self):
        """no lock or counter was lost"""
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertFalse(self.svc.status["busy"])

    def healthy_load(self):
        self.eng.fail_restart = None
        if self.vis is not None:
            self.vis.fail_restart = None
        self.svc.load()
        self.assertTrue(self.eng.alive())
        self.assertClean()


class Loading(Lifecycle):
    def test_state_matrix_what_is_started_and_in_which_order(self):
        for es, vs in itertools.product(("alive", "unloaded", "dead"), (None, "up", "down")):
            with self.subTest(engine=es, vision=vs):
                self.make(es, None if vs is None else {"alive": vs == "up"})
                self.svc.load()
                want = (["vision.restart"] if vs == "down" else []) + (["engine.restart"] if es != "alive" else [])   # the encoder first, as at a start
                self.assertEqual(self.trace, want)
                st = self.state()
                self.assertEqual(st["engine"], "alive")
                self.assertIn(st["vision"], (None, "up"))
                self.assertClean()

    def test_two_concurrent_loads_start_the_engine_once_and_nothing_unloads_it_meanwhile(self):
        self.make("unloaded")
        self.eng.restart_gate = threading.Event()
        errors = []

        def load():
            try:
                self.svc.load()
            except BaseException as e:           # noqa: BLE001
                errors.append(e)
        a = threading.Thread(target=load)
        a.start()
        self.assertTrue(self.eng.restart_entered.wait(DEADLINE))
        b = threading.Thread(target=load)
        b.start()
        deadline = time.monotonic() + DEADLINE
        while self.svc.status["queued"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.svc.status["queued"], 1)                 # the second one is counted as waiting for the turn
        self.assertEqual(self.svc.unload(), "busy")                    # not while the first start runs
        self.eng.restart_gate.set()
        a.join(DEADLINE)
        b.join(DEADLINE)
        self.assertEqual(errors, [])
        self.assertEqual(self.eng.starts, 1)
        self.assertEqual(self.trace, ["engine.restart"])
        self.assertClean()

    def test_a_failed_first_start_fails_every_waiting_load_and_a_healthy_one_works_after(self):
        self.make("unloaded", fail_restart=RuntimeError("boom"))
        self.eng.restart_gate = threading.Event()
        errors = []

        def load():
            try:
                self.svc.load()
            except BaseException as e:           # noqa: BLE001
                errors.append(str(e))
        a = threading.Thread(target=load)
        a.start()
        self.assertTrue(self.eng.restart_entered.wait(DEADLINE))
        b = threading.Thread(target=load)
        b.start()
        deadline = time.monotonic() + DEADLINE
        while self.svc.status["queued"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.eng.restart_gate.set()
        a.join(DEADLINE)
        b.join(DEADLINE)
        self.assertEqual(errors, ["boom", "boom"])
        self.assertEqual(self.trace.count("engine.restart"), 2)        # each tries for itself: no shared failure state
        self.assertClean()
        self.eng.restart_gate = None
        self.healthy_load()
        self.assertEqual(self.eng.starts, 1)

    def test_a_vision_encoder_that_fails_to_start_stops_the_load_before_the_engine(self):
        self.make("unloaded", {"alive": False, "fail_restart": OSError("no encoder")})
        with self.assertRaises(server.EngineDied) as cm:
            self.svc.load()
        self.assertIn("vision encoder failed to start", str(cm.exception))
        self.assertEqual(self.trace, ["vision.restart"])
        self.assertEqual(self.eng.state, "unloaded")
        self.assertClean()
        self.healthy_load()

    def test_a_failed_engine_start_takes_the_encoder_it_started_down_again(self):
        self.make("unloaded", {"alive": False}, fail_restart=RuntimeError("engine boom"))
        with self.assertRaises(RuntimeError) as cm:
            self.svc.load()
        self.assertEqual(str(cm.exception), "engine boom")
        self.assertEqual(self.trace, ["vision.restart", "engine.restart", "vision.unload"])
        self.assertEqual(self.state()["vision"], "down")
        self.assertClean()
        self.healthy_load()
        self.assertEqual(self.state()["vision"], "up")

    def test_a_failing_cleanup_does_not_hide_the_engine_error_and_the_held_flag_is_reset(self):
        """was: the encoder's unload error replaced the engine's, and vision_held stayed True for an encoder that is down"""
        self.make("unloaded", {"alive": False, "fail_unload": RuntimeError("encoder unload boom")}, fail_restart=RuntimeError("engine boom"))
        self.svc.vision_held = True
        with self.assertRaises(RuntimeError) as cm:
            self.svc.load()
        self.assertEqual(str(cm.exception), "engine boom")
        self.assertFalse(self.svc.vision_held)
        self.assertClean()

    def test_the_vision_held_flag_follows_an_encoder_that_a_failed_start_took_down(self):
        self.make("unloaded", {"alive": False}, fail_restart=RuntimeError("engine boom"))
        self.svc.vision_held = True
        with self.assertRaises(RuntimeError):
            self.svc.load()
        self.assertEqual(self.state()["vision"], "down")
        self.assertFalse(self.svc.vision_held)


class BeforeLoadHook(Lifecycle):
    def test_decision_table(self):
        """hook outcome x strict x ignore x min_free_vram_mib x free VRAM (None = unknown) -> loaded / 503 (GpuBusy), hook run once, nothing left locked"""
        for outcome, strict, ignore, minfree, free in itertools.product(["ok", 3, "timeout", "oserror"], [False, True], [False, True], [0, 8000], [None, 2000, 9000]):
            if minfree == 0 and free != 9000:
                continue
            failed = outcome != "ok"
            if failed and strict and not ignore:
                want = "503"
            elif failed and ignore:
                want = "loaded"                                          # ignore always loads, without a VRAM check
            elif minfree and free is not None and free < minfree:
                want = "503"                                             # an unknown reading never refuses
            else:
                want = "loaded"
            with self.subTest(hook=outcome, strict=strict, ignore=ignore, min_free=minfree, free=free):
                self.make("unloaded")
                svc = self.svc
                svc.before_load, svc.before_load_strict, svc.before_load_ignore_failure, svc.min_free_vram_mib = ["hook"], strict, ignore, minfree
                svc.free_vram_mib = lambda free=free: free
                hook = Hook(self.trace, outcome)
                with hook.patch():
                    try:
                        svc.load()
                        got = "loaded"
                    except GpuBusy:
                        got = "503"
                self.assertEqual(got, want)
                self.assertEqual(len(hook.calls), 1)
                self.assertEqual(self.eng.starts, 1 if want == "loaded" else 0)
                self.assertEqual(self.trace, ["hook"] + (["engine.restart"] if want == "loaded" else []))
                self.assertClean()
                svc.before_load_strict = svc.before_load_ignore_failure = False
                svc.min_free_vram_mib = 0
                with Hook(self.trace, "ok").patch():
                    svc.load()                                           # the next, healthy call
                self.assertTrue(self.eng.alive())

    def test_the_hook_gets_the_configured_command_timeout_and_no_stdin(self):
        self.make("unloaded")
        self.svc.before_load, self.svc.before_load_timeout_s = "do-something", 7.0
        hook = Hook(self.trace)
        with hook.patch():
            self.svc.load()
        self.assertEqual(hook.calls, [{"cmd": "do-something", "shell": True, "timeout": 7.0}])
        self.make("unloaded")
        self.svc.before_load = ["prog", "arg"]
        hook = Hook(self.trace)
        with hook.patch():
            self.svc.load()
        self.assertEqual(hook.calls, [{"cmd": ["prog", "arg"], "shell": False, "timeout": 120.0}])

    def test_the_hook_runs_for_a_down_encoder_even_when_the_engine_is_alive_POLICY(self):
        self.make("alive", {"alive": False})
        self.svc.before_load = ["hook"]
        hook = Hook(self.trace)
        with hook.patch():
            self.svc.load()
        self.assertEqual(self.trace, ["hook", "vision.restart"])

    def test_nothing_runs_when_everything_is_up(self):
        self.make("alive", {"alive": True})
        self.svc.before_load = ["hook"]
        hook = Hook(self.trace)
        with hook.patch():
            self.svc.load()
        self.assertEqual(self.trace, [])

    def test_the_hook_holds_the_turn_so_an_unload_during_it_is_busy(self):
        self.make("unloaded")
        self.svc.before_load = ["hook"]
        seen = {}

        class Slow(Hook):
            def __call__(inner, cmd, **kw):
                seen["unload"] = self.svc.unload()
                seen["locked"] = self.svc.fifo.locked()
                return super().__call__(cmd, **kw)
        with Slow(self.trace).patch():
            self.svc.load()
        self.assertEqual(seen, {"unload": "busy", "locked": True})


class Unloading(Lifecycle):
    def test_idle_busy_queued_and_recent(self):
        self.make("alive", {"alive": True})
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.trace, ["engine.unload", "vision.unload"])
        self.assertEqual((self.state()["engine"], self.state()["vision"], self.svc.vision_held), ("unloaded", "down", False))
        self.assertEqual(self.svc.unload(), "not loaded")
        for label, setup in (("busy", lambda s: s.status.update(busy=True)), ("queued", lambda s: s.status.update(queued=1)),
                             ("recent", lambda s: setattr(s, "last_request_at", time.time()))):
            with self.subTest(label):
                self.make("alive")
                setup(self.svc)
                self.assertEqual(self.svc.unload(idle_for=60) if label == "recent" else self.svc.unload(), "busy")
                self.assertEqual(self.trace, [])
                self.assertEqual(self.eng.state, "alive")
        self.make("alive")
        with self.svc.fifo:
            self.assertEqual(self.svc.unload(), "busy")
        self.assertEqual(self.trace, [])

    def test_an_engine_without_unload_is_unsupported(self):
        self.make("alive")
        class Plain:
            max_context = 4096
            def alive(self): return True
        self.svc.engine = Plain()
        self.assertEqual(self.svc.unload(), "unsupported")

    def test_an_engine_that_cannot_stop_leaves_everything_as_it_was_and_the_next_try_works(self):
        self.make("alive", {"alive": True}, fail_unload=EngineStuck("stuck"))
        with self.assertRaises(EngineStuck):
            self.svc.unload()
        self.assertEqual(self.trace, ["engine.unload"])                 # the encoder is not touched
        self.assertEqual(self.state()["vision"], "up")
        self.assertClean()
        self.eng.fail_unload = None
        self.assertEqual(self.svc.unload(), "unloaded")

    def test_an_engine_that_is_gone_does_not_keep_its_encoder_holding_the_GPU(self):
        """was: "not loaded" and the encoder stayed up - also the second try after a failed encoder unload"""
        self.make("dead", {"alive": True})
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.state()["vision"], "down")
        self.assertEqual(self.trace, ["vision.unload"])
        self.assertFalse(self.svc.vision_held)
        self.assertEqual(self.svc.unload(), "not loaded")               # nothing left to stop
        self.assertClean()

    def test_a_dead_engine_with_a_live_encoder_is_still_busy_while_a_request_is_recorded(self):
        self.make("dead", {"alive": True})
        self.svc.status["queued"] = 1
        self.assertEqual(self.svc.unload(), "busy")
        self.assertEqual(self.state()["vision"], "up")
        with self.svc.fifo:
            self.assertEqual(self.svc.unload(), "busy")

    def test_a_dead_engine_without_an_encoder_is_not_loaded(self):
        self.make("dead")
        self.assertEqual(self.svc.unload(), "not loaded")
        self.make("dead", {"alive": False})
        self.assertEqual(self.svc.unload(), "not loaded")
        self.assertEqual(self.trace, [])

    def test_a_failed_encoder_unload_can_be_repeated(self):
        self.make("alive", {"alive": True, "fail_unload": RuntimeError("encoder boom")})
        with self.assertRaises(RuntimeError):
            self.svc.unload()
        self.assertEqual((self.state()["engine"], self.state()["vision"]), ("unloaded", "up"))
        self.assertClean()
        self.vis.fail_unload = None
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.state()["vision"], "down")
        self.healthy_load()

    def test_a_request_between_load_and_its_turn_is_not_counted_so_a_late_idle_unload_takes_the_engine_POLICY(self):
        """the handler's load() only stamps last_request_at; the request is counted (queued) when it reaches its turn.  A long prepare (a big
        prompt, a picture) longer than idle_unload_s lets the idle thread unload the engine in between; the request then loads it again."""
        self.make("alive")
        now = [1000.0]
        with mock.patch.object(server.time, "time", lambda: now[0]):
            self.svc.load()
            now[0] += 5.0
            self.assertEqual(self.svc.unload(idle_for=2.0), "unloaded")
        self.assertEqual(self.svc.status["queued"], 0)
        self.healthy_load()


class Waiting(Lifecycle):
    def clocked(self, results):
        self.make("alive")
        clock, sleeps = [0.0], []
        seq = iter(results)
        self.svc.unload = lambda idle_for=None: next(seq)
        return clock, sleeps, mock.patch.object(server.time, "monotonic", lambda: clock[0]), \
            mock.patch.object(server.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))

    def test_it_polls_until_free(self):
        clock, sleeps, m1, m2 = self.clocked(["busy", "busy", "busy", "unloaded"])
        with m1, m2:
            self.assertEqual(self.svc.unload_waiting(2.0), "unloaded")
        self.assertEqual(sleeps, [0.25, 0.25, 0.25])

    def test_it_gives_up_at_the_deadline(self):
        clock, sleeps, m1, m2 = self.clocked(["busy"] * 50)
        with m1, m2:
            self.assertEqual(self.svc.unload_waiting(1.0), "busy")
        self.assertEqual(len(sleeps), 4)

    def test_zero_and_negative_waits_ask_once(self):
        for w in (0.0, -5.0):
            clock, sleeps, m1, m2 = self.clocked(["busy"] * 5)
            with m1, m2:
                self.assertEqual(self.svc.unload_waiting(w), "busy")
            self.assertEqual(sleeps, [])

    def test_a_non_busy_answer_ends_it_at_once(self):
        for answer in ("not loaded", "unsupported"):
            clock, sleeps, m1, m2 = self.clocked([answer])
            with m1, m2:
                self.assertEqual(self.svc.unload_waiting(5.0), answer)
            self.assertEqual(sleeps, [])


class Stop(BaseException):
    pass


class IdleLoop(Lifecycle):
    def run_loop(self, first_error, rounds=4):
        self.make("alive")
        self.svc.idle_unload_s = 40
        n = {"unload": 0, "sleep": 0}

        def unload(idle_for=None):
            n["unload"] += 1
            if n["unload"] == 1 and first_error is not None:
                raise first_error
            return "busy"

        def sleep(_s):
            n["sleep"] += 1
            if n["sleep"] > rounds:
                raise Stop()
        died = []
        old = threading.excepthook
        threading.excepthook = lambda a: died.append(a.exc_type)
        self.svc.unload = unload
        try:
            with mock.patch.object(server.time, "sleep", sleep):
                self.svc.start_idle_unload()
                deadline = time.monotonic() + DEADLINE
                while n["sleep"] <= rounds and time.monotonic() < deadline:
                    threading.Event().wait(0.01)
                threading.Event().wait(0.1)
        finally:
            threading.excepthook = old
        return n, [d for d in died if d is not Stop]

    def test_it_keeps_going_after_an_engine_that_cannot_stop(self):
        n, died = self.run_loop(EngineStuck("stuck"))
        self.assertEqual((n["unload"], died), (4, []))

    def test_it_keeps_going_after_any_other_unload_error(self):
        """was: a RuntimeError / OSError from engine.unload or the encoder's unload ended the thread and idle unloading stopped for good"""
        for exc in (RuntimeError("encoder boom"), OSError("kill failed")):
            with self.subTest(exc=repr(exc)):
                n, died = self.run_loop(exc)
                self.assertEqual(died, [])
                self.assertEqual(n["unload"], 4)

    def test_it_does_nothing_when_idle_unload_is_off(self):
        self.make("alive")
        self.svc.idle_unload_s = 0
        with mock.patch.object(threading, "Thread") as thread:
            self.svc.start_idle_unload()
        thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
