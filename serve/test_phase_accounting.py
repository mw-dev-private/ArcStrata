"""serve/test_phase_accounting.py - what each phase and each token figure of a request measures (A12b).

Part 1 drives Service.load / run with a fake clock (server.time is a proxy: time(), perf_counter() and monotonic() read one clock that only the fake
engine, a fake restart or the test advances; sleep() advances it): the monitor record's queue_s / load_s / first_token_s, the history row's duration_s
and the latency histograms are checked against what the clock was told.  Part 2 sends one request through chat / messages / responses against an
engine that sets chosen DONE figures and compares API usage, timings, the history row, the totals and both Prometheus dialects with an explicit
balance.  No hardware, process or real wait; Service.start_telemetry is switched off before any Service exists.

Tests named ..._POLICY pin a measurement definition that differs between two places or from its name; they exist so a change is noticed, not
because it is endorsed."""
import contextlib
import http.client
import io
import json
import re
import threading
import time as real_time
import unittest
from pathlib import Path
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve


class Clock:
    def __init__(self, t0=1_000_000.0):
        self.now = t0

    def advance(self, dt):
        self.now += dt


class TimeProxy:
    def __init__(self, clock):
        self._c = clock

    def time(self):
        return self._c.now

    def perf_counter(self):
        return self._c.now

    def monotonic(self):
        return self._c.now

    def sleep(self, s):
        self._c.advance(s)

    def __getattr__(self, name):
        return getattr(real_time, name)


class Timed(MockEngine):
    """advances the fake clock while generating (first_delay before the first token, step per token) and sets the DONE figures when closed"""

    def __init__(self, clock, text="hello", first_delay=1.5, step=0.1, state="alive", **figs):
        super().__init__(ByteTokenizer(), text, max_context=1 << 16)
        self.clock, self.first_delay, self.step, self.state, self.figs = clock, first_delay, step, state, figs
        self.last, self.restart_s, self.starts, self.on_restart = {}, 4.0, 0, None

    @property
    def unloaded(self):
        return self.state == "unloaded"

    def alive(self):
        return self.state == "alive"

    def exit_code(self):
        return 9

    def unload(self):
        self.state = "unloaded"

    def restart(self):
        if self.on_restart:
            self.on_restart()
        self.clock.advance(self.restart_s)
        self.state = "alive"
        self.starts += 1

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        k = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                self.clock.advance(self.first_delay if k == 0 else self.step)
                k += 1
                yield t
        finally:
            f = dict(self.figs)
            f.setdefault("generated", k)
            f.setdefault("prompt_tokens", len(ids))
            f.setdefault("finish", "stop")
            self.last = f


class Phases(unittest.TestCase):
    def make(self, **kw):
        Service.start_telemetry = lambda self: None
        self.clock = Clock()
        patcher = mock.patch.object(server, "time", TimeProxy(self.clock))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.eng = Timed(self.clock, **kw)
        self.svc = Service(self.eng, ByteTokenizer(), ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True
        return self.svc

    def request(self, prepare_s=0.0, ids_len=40, hold_turn_s=None):
        svc = self.svc
        rec = svc.begin_request("/v1/chat/completions", {"model": "m"})
        svc.load()
        if prepare_s:
            self.clock.advance(prepare_s)
        gen = svc.run(list(range(65, 65 + ids_len)), False, None, 60, {}, threading.Event())
        if hold_turn_s is not None:                              # the turn is busy for this long: the other request ends meanwhile
            svc.fifo.acquire()
            t = threading.Thread(target=lambda: (real_time.sleep(0.2), self.clock.advance(hold_turn_s), svc.fifo.release()))
            t.start()
            self.addCleanup(t.join)
        for _ in gen:
            pass
        return rec

    def hist(self):
        return self.svc.history[-1]

    def hist_sum(self, name):
        return self.svc.latencies.snapshot()[name][1]

    def test_the_phases_of_a_plain_request_and_what_each_clock_covers(self):
        """prepare 2 s, first token after 1.5 s, 10 tokens 0.1 s apart (the first at +1.5): first_token_s counts from the request's arrival, the
        history row and the latency histograms count from the moment the request holds the turn and is registered running"""
        self.make(text="0123456789")
        rec = self.request(prepare_s=2.0)
        self.assertEqual((rec["queue_s"], rec["load_s"]), (0.0, 0.0))
        self.assertEqual(rec["first_token_s"], 3.5)                 # prepare 2.0 + prompt read 1.5: arrival -> the engine's first token
        self.assertEqual(self.hist()["duration_s"], 2.5)            # prompt read 1.5 + decode 0.9 + the end marker's 0.1
        self.assertAlmostEqual(self.hist_sum("ttft"), 1.5)          # the engine's first token, not the first visible character
        self.assertAlmostEqual(self.hist_sum("e2e"), 2.5)           # NOT the whole request: prepare, queue and load are outside
        self.assertAlmostEqual(self.clock.now - rec["started_at"], 4.5)   # the whole request took longer than any recorded phase

    def test_queue_wait_is_the_other_requests_turn(self):
        self.make(text="01234")
        rec = self.request(hold_turn_s=3.0)
        self.assertEqual(rec["queue_s"], 3.0)
        self.assertEqual(rec["load_s"], 0.0)
        self.assertEqual(rec["first_token_s"], 4.5)                 # wait 3.0 + prompt read 1.5

    def test_a_start_inside_load_is_load_time(self):
        self.make(text="01234", state="unloaded")
        rec = self.request()
        self.assertEqual((rec["queue_s"], rec["load_s"]), (0.0, 4.0))
        self.assertEqual(rec["first_token_s"], 5.5)
        self.assertAlmostEqual(self.hist_sum("ttft"), 1.5)          # the engine's TTFT does not contain the start

    def test_a_start_at_the_requests_own_turn_is_load_time_too(self):
        """was: the request before it killed the engine; this request's own turn restarted it (4 s) and load_s stayed 0 while the record
        said 'generating' - the 4 s were in no phase except first_token_s"""
        self.make(text="01234")
        rec = self.svc.begin_request("/v1/chat/completions", {"model": "m"})
        self.svc.load()                                              # the engine is alive here
        self.eng.state = "dead"                                      # ... and gone before this request's turn
        seen = []
        self.eng.on_restart = lambda: seen.append(rec["state"])
        for _ in self.svc.run(list(range(65, 105)), False, None, 60, {}, threading.Event()):
            pass
        self.assertEqual(self.eng.starts, 1)
        self.assertEqual(seen, ["loading"])                          # the label says what is happening
        self.assertEqual(rec["load_s"], 4.0)
        self.assertEqual(rec["queue_s"], 0.0)
        self.assertEqual(rec["state"], "generating")
        self.assertEqual(rec["first_token_s"], 5.5)

    def test_a_request_that_needs_no_start_leaves_load_s_alone(self):
        self.make(text="01234")
        rec = self.request()
        self.assertEqual(rec["load_s"], 0.0)

    def test_a_failed_start_at_the_turn_is_still_counted_and_does_not_leave_the_label_behind(self):
        self.make(text="01234")
        rec = self.svc.begin_request("/v1/chat/completions", {"model": "m"})
        self.svc.load()
        self.eng.state = "dead"
        self.eng.restart = lambda: (self.clock.advance(2.0), (_ for _ in ()).throw(RuntimeError("no engine")))[1]
        with self.assertRaises(RuntimeError):
            for _ in self.svc.run(list(range(65, 105)), False, None, 60, {}, threading.Event()):
                pass
        self.assertEqual(rec["load_s"], 2.0)
        self.assertEqual(rec["state"], "generating")
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertFalse(self.svc.fifo.locked())


class BatchFifo(Phases):
    """with batch slots a request does not hold the single turn; it takes the control lock only to start the engine: the wait for that lock is queue
    time, only the start itself is load time (r02: the first version counted the wait as load)"""

    def make_batch(self, **kw):
        self.make(**kw)
        self.eng.batch = 2
        return self.svc

    def run_batch(self, other_starts, fail=None):
        """a request arrives with the engine gone; another holder of the lock keeps it for 3 s (and starts the engine meanwhile when other_starts)"""
        svc, eng = self.svc, self.eng
        rec = svc.begin_request("/v1/chat/completions", {"model": "m"})
        eng.state = "dead"
        states = []
        eng.on_restart = lambda: states.append(("at start", rec["state"]))
        real_ensure = svc.ensure_loaded
        svc.ensure_loaded = lambda: (states.append(("at ensure_loaded", rec["state"])), real_ensure())[1]
        if fail:
            eng.restart = lambda: (self.clock.advance(2.0), (_ for _ in ()).throw(fail))[1]
        class OtherHolder:
            """the lock as seen by a request that has to wait for another holder: when it asks, the other holder has had it for 3 s and gives it
            up (the fake clock moves); optionally the other holder has started the engine meanwhile"""

            def __init__(inner):
                inner.real = threading.Lock()

            def acquire(inner, *a, **k):
                states.append(("while waiting", rec["state"]))
                self.clock.advance(3.0)
                if other_starts:
                    eng.state = "alive"
                    eng.starts += 1
                return inner.real.acquire(*a, **k)

            def release(inner):
                inner.real.release()

            def __enter__(inner):
                inner.acquire()
                return inner

            def __exit__(inner, *exc):
                inner.release()

            def locked(inner):
                return inner.real.locked()
        svc.fifo = OtherHolder()
        err = None
        try:
            for _ in svc.run(list(range(65, 105)), False, None, 60, {}, threading.Event()):
                pass
        except Exception as e:           # noqa: BLE001
            err = e
        return rec, states, err

    def test_the_wait_is_queue_time_and_the_own_start_is_load_time(self):
        self.make_batch(text="01234")
        rec, states, err = self.run_batch(other_starts=False)
        self.assertIsNone(err)
        self.assertEqual((rec["queue_s"], rec["load_s"]), (3.0, 4.0))               # was (0.0, 7.0)
        self.assertEqual(states, [("while waiting", "queued"), ("at ensure_loaded", "loading"), ("at start", "loading")])
        self.assertEqual(rec["state"], "generating")
        self.assertEqual(self.eng.starts, 1)
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertFalse(self.svc.fifo.locked())

    def test_an_engine_another_request_started_meanwhile_costs_this_one_no_load_time(self):
        self.make_batch(text="01234")
        rec, states, err = self.run_batch(other_starts=True)
        self.assertIsNone(err)
        self.assertEqual((rec["queue_s"], rec["load_s"]), (3.0, 0.0))               # was (0.0, 3.0) with 0 starts of its own
        self.assertEqual(self.eng.starts, 1)                                         # the other one's start, no second one
        self.assertEqual(states, [("while waiting", "queued"), ("at ensure_loaded", "generating")])   # a start is not needed: it never says "loading"
        self.assertEqual(rec["state"], "generating")

    def test_a_failed_start_counts_only_the_start_and_leaves_no_label_lock_or_counter(self):
        self.make_batch(text="01234")
        rec, states, err = self.run_batch(other_starts=False, fail=RuntimeError("no engine"))
        self.assertIsInstance(err, RuntimeError)
        self.assertEqual((rec["queue_s"], rec["load_s"]), (3.0, 2.0))
        self.assertEqual(rec["state"], "generating")
        self.assertEqual(self.svc.status["queued"], 0)
        self.assertFalse(self.svc.fifo.locked())

    def test_the_two_do_not_overlap(self):
        self.make_batch(text="01234")
        rec, _, _ = self.run_batch(other_starts=False)
        self.assertEqual(rec["queue_s"] + rec["load_s"], 7.0)                        # the 3 s wait and the 4 s start, each once
        self.assertEqual(rec["first_token_s"], 8.5)                                  # + the 1.5 s prompt read, from arrival

    def test_a_loaded_engine_takes_no_lock_and_no_time(self):
        self.make_batch(text="01234")
        rec = self.request()
        self.assertEqual((rec["queue_s"], rec["load_s"]), (0.0, 0.0))


class Figure(MockEngine):
    """sets chosen DONE figures when the generator is closed (as StrataEngine does after draining the DONE line); figs=None: no DONE at all"""

    def __init__(self, tok):
        super().__init__(tok, "hello there", max_context=1 << 16)
        self.figs, self.fail_after, self.cancel_after, self.last, self.last_prompt = {}, None, None, {}, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        k = 0
        died = False
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                k += 1
                if self.fail_after is not None and k > self.fail_after:
                    died = True
                    raise EngineDied("the engine ended mid-answer (simulated)")
                if self.cancel_after is not None and k > self.cancel_after:
                    cancel.set()
                    return
                yield t
        finally:
            if self.figs is not None and not died:
                f = dict(self.figs)
                f.setdefault("generated", k)
                f.setdefault("prompt_tokens", len(ids))
                self.last = f


ROUTES = {
    "chat": ("/v1/chat/completions", lambda: {"model": "m", "messages": [{"role": "user", "content": "say hello"}], "max_tokens": 200, "reasoning_effort": "none"}),
    "chat-stream": ("/v1/chat/completions", lambda: {"model": "m", "messages": [{"role": "user", "content": "say hello"}], "max_tokens": 200, "reasoning_effort": "none",
                                                      "stream": True, "stream_options": {"include_usage": True}}),
    "messages": ("/v1/messages", lambda: {"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": "say hello"}]}),
    "responses": ("/v1/responses", lambda: {"model": "m", "input": "say hello", "max_output_tokens": 200}),
}


class Balance(unittest.TestCase):
    def setUp(self):
        Service.start_telemetry = lambda self: None
        tok = ByteTokenizer()
        self.eng = Figure(tok)
        self.svc = Service(self.eng, tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]
        self.addCleanup(lambda: (self.httpd.shutdown(), self.httpd.server_close()))

    def call(self, method, path, body=None, accept=None):
        with contextlib.redirect_stdout(io.StringIO()):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
            try:
                c.request(method, path, body=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01", **({"Accept": accept} if accept else {})})
                r = c.getresponse()
                return r.status, r.read().decode()
            except (http.client.RemoteDisconnected, ConnectionError) as e:
                return None, type(e).__name__
            finally:
                c.close()

    def clean(self):
        with self.svc.status_lock:
            self.svc.history.clear()
            for k, v in list(self.svc.totals.items()):
                if k != "since" and isinstance(v, (int, float)):
                    self.svc.totals[k] = 0
        self.svc.latencies.__init__()

    def run_one(self, route, figs, fail_after=None, cancel_after=None):
        """-> dict(usage, timings, history, totals, prom): one request; figures named with a negative number are relative to the prompt length"""
        path, mk = ROUTES[route]
        self.eng.figs = None
        self.call("POST", path, mk())                               # learn the prompt length
        length = len(self.eng.last_prompt)
        if figs is not None:
            figs = {k: (max(0, length + v) if isinstance(v, int) and v < 0 and k in ("prompt_read", "reused") else v) for k, v in figs.items()}
        self.clean()
        self.eng.figs, self.eng.fail_after, self.eng.cancel_after = figs, fail_after, cancel_after
        status, raw = self.call("POST", path, mk())
        real_time.sleep(0.3)
        metrics = json.loads(self.call("GET", "/metrics")[1])
        out = {"status": status, "length": length, "history": metrics["requests"][0] if metrics.get("requests") else None, "totals": metrics["totals"]}
        usage = timings = None
        try:
            if route == "chat-stream":
                ev = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: {")]
                usage = next((e["usage"] for e in reversed(ev) if e.get("usage")), None)
                timings = next((e["timings"] for e in reversed(ev) if e.get("timings")), None)
            else:
                j = json.loads(raw)
                usage, timings = j.get("usage"), j.get("timings")
        except ValueError:
            pass
        out["usage"], out["timings"] = usage, timings
        vllm = self.call("GET", "/metrics", accept="text/plain")[1]
        llama = self.call("GET", "/metrics/prometheus")[1]
        grab = lambda text, pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, text, re.M))
        out["prom"] = {"hits": grab(vllm, r"^vllm:prefix_cache_hits_total\{[^}]*\} (\S+)"), "queries": grab(vllm, r"^vllm:prefix_cache_queries_total\{[^}]*\} (\S+)"),
                       "success": grab(vllm, r"^vllm:request_success_total\{[^}]*\} (\S+)"), "generated": grab(vllm, r"^vllm:generation_tokens_total\{[^}]*\} (\S+)"),
                       "ll_prompt": grab(llama, r"^llamacpp:prompt_tokens_total (\S+)"), "ll_cached": grab(llama, r"^llamacpp:prompt_tokens_cached_total (\S+)"),
                       "ll_predicted": grab(llama, r"^llamacpp:tokens_predicted_total (\S+)")}
        return out

    @staticmethod
    def api_numbers(route, usage):
        if route.startswith("messages"):
            return usage["input_tokens"] + usage.get("cache_read_input_tokens", 0), usage["output_tokens"], usage.get("cache_read_input_tokens", 0)
        if route == "responses":
            return usage["input_tokens"], usage["output_tokens"], usage["input_tokens_details"]["cached_tokens"]
        return usage["prompt_tokens"], usage["completion_tokens"], usage["prompt_tokens_details"]["cached_tokens"]

    def test_the_balance_of_a_regular_request_on_every_route(self):
        """full prompt L, conversation cache reused 5, the engine says prompt_read = L - 5: every consumer says the same four numbers"""
        for route in ROUTES:
            with self.subTest(route=route):
                r = self.run_one(route, dict(reused=5, prompt_read=-5, prompt_ms=40.0, decode_ms=200.0, finish="stop"))
                L = r["length"]
                prompt, completion, cached = self.api_numbers(route, r["usage"])
                self.assertEqual((prompt, cached), (L, 5))
                h, t = r["history"], r["totals"]
                self.assertEqual((h["prompt_tokens"], h["reused"], h["prompt_read"], h["prompt_total"]), (L, 5, L - 5, L))
                self.assertEqual((h["output_tokens"], h["engine_generated"]), (completion, completion))      # nothing else was generated here
                self.assertEqual((t["requests"], t["prompt_tokens"], t["reused"], t["output_tokens"]), (1, L, 5, completion))
                self.assertEqual((r["prom"]["hits"], r["prom"]["queries"], r["prom"]["generated"], r["prom"]["ll_prompt"], r["prom"]["ll_cached"]), (5.0, float(L), float(completion), float(L - 5), 5.0))
                if r["timings"] is not None:
                    self.assertEqual((r["timings"]["cache_n"], r["timings"]["prompt_n"], r["timings"]["predicted_n"]), (5, L - 5, completion))

    def test_an_old_engine_without_reused_and_prompt_read_says_zero_cached_and_the_whole_prompt(self):
        for route in ROUTES:
            with self.subTest(route=route):
                r = self.run_one(route, dict(prompt_ms=40.0, decode_ms=200.0, finish="stop"))
                L = r["length"]
                prompt, completion, cached = self.api_numbers(route, r["usage"])
                self.assertEqual((prompt, cached), (L, 0))
                h = r["history"]
                self.assertEqual((h["reused"], h["prompt_read"]), (None, None))            # unknown stays unknown in the row ...
                self.assertEqual(r["totals"]["reused"], 0)                                   # ... and counts 0 in the totals
                self.assertEqual((r["prom"]["ll_prompt"], r["prom"]["ll_cached"]), (float(L), 0.0))

    def test_the_slot_session_case_the_api_and_the_totals_disagree_POLICY(self):
        """#1870: the slot's own session held the prompt (reused 0, prompt_read 0): the API reports it as cached (cache_n = L), the history row keeps
        reused 0 / prompt_read 0, and the totals and both Prometheus dialects count it as READ (llamacpp:prompt_tokens_total = L) and as no cache hit.
        Two places answer the same question with different numbers; which one is meant is a definition for the owner (see the report)."""
        for route in ROUTES:
            with self.subTest(route=route):
                r = self.run_one(route, dict(reused=0, prompt_read=0, prompt_ms=1.0, decode_ms=200.0, finish="stop"))
                L = r["length"]
                _, _, cached = self.api_numbers(route, r["usage"])
                self.assertEqual(cached, L - 1 if route == "messages" else L)       # (Anthropic's input_tokens is never below 1)
                self.assertEqual((r["history"]["reused"], r["history"]["prompt_read"]), (0, 0))
                self.assertEqual(r["totals"]["reused"], 0)
                self.assertEqual((r["prom"]["hits"], r["prom"]["ll_cached"], r["prom"]["ll_prompt"]), (0.0, 0.0, float(L)))

    def test_a_prompt_read_cut_short_by_a_cancel_counts_what_was_read(self):
        r = self.run_one("chat", dict(reused=2, prompt_read=-20, prompt_ms=30.0, decode_ms=5.0, finish="cancel"), cancel_after=4)
        L = r["length"]
        h, t = r["history"], r["totals"]
        seen = 2 + (L - 20)
        self.assertEqual((h["finish"], h["prompt_tokens"], h["prompt_total"], h["prompt_read"]), ("cancel", seen, L, L - 20))
        self.assertEqual((t["prompt_tokens"], t["reused"]), (seen, 2))
        self.assertEqual(r["prom"]["ll_prompt"], float(seen - 2))

    def test_an_answer_without_a_done_line_is_counted_with_the_whole_prompt_and_unknown_engine_figures_POLICY(self):
        for label, kw in (("silent end", {}), ("engine died after 3 tokens", {"fail_after": 3})):
            with self.subTest(label):
                r = self.run_one("chat", None, **kw)
                h, t = r["history"], r["totals"]
                self.assertEqual(h["engine_generated"], None)                                # not invented
                self.assertEqual(h["prompt_tokens"], r["length"])                            # the prompt is counted whole although no DONE said it was read
                self.assertEqual(t["requests"], 1)
                self.assertEqual(r["prom"]["success"], 1.0)                                  # 'success' counts the request that died too
                if kw:
                    self.assertEqual(h["finish"], "error")
                    self.assertEqual(h["output_tokens"], 3)
                    self.assertEqual(t["output_tokens"], 3)

    def test_real_zero_and_unknown_in_the_timings(self):
        from serve.server import request_timings
        zero_decode = request_timings(40, 12, {"prompt_ms": 10.0, "decode_ms": 0.0, "generated": 12, "reused": 3, "finish": "stop"})
        self.assertEqual((zero_decode["predicted_ms"], zero_decode["predicted_per_second"], zero_decode["predicted_per_token_ms"]), (0.0, None, 0.0))
        all_cached = request_timings(40, 12, {"prompt_ms": 0.0, "decode_ms": 100.0, "generated": 12, "reused": 39, "prompt_read": 0, "finish": "stop"})
        self.assertEqual((all_cached["cache_n"], all_cached["prompt_n"], all_cached["prompt_per_second"], all_cached["prompt_per_token_ms"]), (40, 0, None, None))
        none = request_timings(40, 12, {"prompt_ms": None})
        self.assertIsNone(none)                                                              # an engine that keeps no clock: no timings at all, not zeros
        no_draft = request_timings(40, 12, {"prompt_ms": 5.0, "decode_ms": 50.0, "generated": 12})
        self.assertNotIn("draft_n", no_draft)
        zero_draft = request_timings(40, 12, {"prompt_ms": 5.0, "decode_ms": 50.0, "generated": 12, "drafts_offered": 0, "drafts_accepted": 0})
        self.assertEqual((zero_draft["draft_n"], zero_draft["draft_n_accepted"]), (0, 0))   # a real zero is kept

    def test_the_engine_count_gives_the_rate_but_the_servers_count_is_predicted_n(self):
        from serve.server import request_timings
        t = request_timings(40, 20, {"prompt_ms": 5.0, "decode_ms": 100.0, "generated": 10, "finish": "stop"})    # 10 native + wrap-up tokens added by the server
        self.assertEqual(t["predicted_n"], 20)
        self.assertEqual(t["predicted_per_second"], 100.0)                                   # 10 / 0.1 s: the engine's own count


if __name__ == "__main__":
    unittest.main()
