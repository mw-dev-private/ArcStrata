"""A03h - DONE lines carry only what the NATIVE emitter can print, and the figures land atomically.

Contract (fixed BEFORE the tests; decisions marked): the single DONE emitter (CPU generate.cpp:
14620, SYCL:11872 - one line for solo AND batch legs) prints `DONE %lld %lld %.1f %.1f %s` plus the
appended optional groups - drafts(2) [0.1.3+], hits/lookups, expert tiers [.31+], prompt_read [.36+],
offloaded [.39+], the --lookup-chain 5-field group, then the named kv fields chunk=/experts_streamed=
(SYCL currently prints no kv group at all).  Every counter is an int64_t delta >= 0 (so %lld's
printable range [0, 2**63-1] bounds it - grammar, the A03g-r02 precedent), every ms/MB is a finite
non-negative %.1f, and `finish` is one of the four words the emitter hard-codes:
stop | length | cancel | nonfinite (generate.cpp:11000/13515/13540/14061/14376; = _BDONE_FINISH).

_parse_done's disposition (this package):
* NEVER fabricate: unicode digits / '+7' / '007' / beyond-%lld / inf / nan are NOT emitter words -
  they count as malformed, exactly like 'x' counts today.  A figure the engine never produced never
  enters self.last (it reaches client timings via request_stats, the /status history, and the
  internal stop/length/cancel/nonfinite branches - #879's retry, carry legs, prompt accounting).
* NEVER a half update: self.last is replaced by ONE fully validated dict or not at all (the parent
  assigned the core first and then died in an optional group's int() - a partial core-only dict
  survived AND the request died).
* OPTIONAL group lying: reading a fully PRESENT but lying group as ABSENT is a DELIBERATE
  contract choice (owner confirmed r02) - it CHANGES the parent's error contract (the parent's
  int() raised there) and is NOT the identical wire state of an older, shorter line (an old
  engine never printed the group; this one printed it wrong).  Consequences, pinned in
  OptionalGroupAccounting: reused -> None (carry merge and /status last_reused skip it),
  prompt_read -> None (prompt accounting falls back to the FULL prompt, #471), hits/lookups ->
  None (history hit_rate None).  Never invented figures, never a half group.
* Malformed MANDATORY core raises StreamCorrupted (A03f family) - OWNER decision 22.10.
  (codex-A03h-owner-rework-confirmed-20261022): an end line the emitter cannot print is an
  ENGINE fault: 503 server_error before the answer starts, an API error event after it started.
  Never the 400 handler, never a token.  LIFECYCLE around that fault (r02, both Codex
  counterexamples pinned in DoneCallerCleanup): DONE is the END marker BEFORE its figures are
  read - the solo reader never STOP-drains onto an end it already consumed (r01: EngineSilent
  replaced the parse fault); a drain (solo STOP drain, _drain_control) reads a lying end line
  as end-without-figures and NEVER raises (r01: the raising drain skipped ctl.release() and
  held the control lock; the pre-existing hole reached even parent-'x' records).
  DELIBERATE degradation (r02, pinned): if the answer had ALREADY completed locally (EOS/stop/
  max_new break) and the DONE only syncs the wire inside the close-drain, the lying stats line
  does NOT kill the finished answer - the request completes with the server's true counts, the
  wire figures read as absent, one log line (parent AND r01 killed a complete reply with a 400).

Fixtures: FAKE_BATCH opt-in `--done-bad sup|arabic|huge|infms|finish|trunc|group9|kvsup|kvarabic`
(one corrupted DONE per run; defaults byte-identical).  Pins green on the parent by design: the
valid 5/9/15/16-field historical prefixes, the chain group's positional skip, and the kv try/except
reading - so the fix may not quietly change those.
"""
import json
import queue
import threading
import time
import types
import unittest
import urllib.error
import urllib.request

from serve import server


class DoneParse(unittest.TestCase):
    """_parse_done over scripted lines: never raises on optional junk, never fabricates, atomic."""

    def engine(self):
        e = object.__new__(server.StrataEngine)
        e.batch = 0                       # plain _last behind the property (no thread-local needed)
        e.last = {}
        return e

    def feed(self, line):
        e = self.engine()
        e._parse_done(line)
        return e

    CORE = "3 60 1200.0 500.0 stop"        # generated, prompt_tokens, prompt_ms, decode_ms, finish
    FULL9 = "3 60 1200.0 500.0 stop 7 9 2"
    FULL15 = "3 60 1200.0 500.0 stop 7 9 2 11 13 2 4 1.5 58 6"

    def test_valid_and_historical_prefixes_stay_exact(self):
        e = self.feed("DONE " + self.CORE)
        self.assertEqual(e.last, {"generated": 3, "prompt_tokens": 60, "prompt_ms": 1200.0,
                                  "decode_ms": 500.0, "finish": "stop"})
        e = self.feed("DONE " + self.FULL9)
        self.assertEqual((e.last["drafts_accepted"], e.last["drafts_offered"], e.last["reused"]),
                         (7, 9, 2), "the 0.1.3+ group")
        e = self.feed("DONE " + self.FULL15)
        self.assertEqual((e.last["hits"], e.last["lookups"], e.last["ram_blobs"], e.last["file_blobs"],
                          e.last["file_mb"], e.last["prompt_read"], e.last["offloaded"]),
                         (11, 13, 2, 4, 1.5, 58, 6), "groups through #588")
        e = self.feed("DONE " + self.FULL15 + " chunk=40 experts_streamed=3")
        self.assertEqual((e.last["chunk"], e.last["experts_streamed"]), (40, 3), "the named kv group")
        e = self.feed("DONE " + self.FULL15 + " 1 2 3 4 5 chunk=40 experts_streamed=3")
        self.assertEqual(e.last.get("chunk"), 40,
                         "the --lookup-chain 5-field group: positional, read as today = skipped, kv still lands")

    def test_beyond_ll_or_nonfinite_never_fabricates(self):
        CASES = [                                     # (line, must NOT contain a fabricated value)
            ("DONE \u00b2 60 1.0 1.0 stop 0 0 0", "parent raised a bare ValueError; r02 makes it StreamCorrupted"),
            ("DONE \u0667 60 1.0 1.0 stop 0 0 0", "int() PARSES today (7): fabrication ends"),
            ("DONE +3 60 1.0 1.0 stop 0 0 0", "%lld prints no plus"),
            ("DONE 003 60 1.0 1.0 stop 0 0 0", "%lld prints no leading zero"),
            ("DONE " + "9" * 1000 + " 60 1.0 1.0 stop 0 0 0", "beyond %lld"),
            ("DONE 3 60 inf 1.0 stop 0 0 0", "float() PARSES inf today"),
            ("DONE 3 60 nan 1.0 stop 0 0 0", "float() PARSES nan today"),
            ("DONE 3 60 \u0667.0 1.0 stop 0 0 0", "float() reads unicode figures today"),
            ("DONE 3 60 -1.0 1.0 stop 0 0 0", "negative: no clock delta prints one"),
            ("DONE 3 60 1.0 1.0 stopped 0 0 0", "finish outside the emitter's four words"),
            ("DONE 3 60 1.0 1.0 \u00b2 0 0 0", "finish junk"),
        ]
        for line, why in CASES:
            with self.subTest(line=line):
                e = self.engine()
                e.last = {"generated": 1, "note": "previous leg"}   # parent: half-updates or fabricates
                with self.assertRaises(server.StreamCorrupted):      # r02 owner policy: engine fault
                    e._parse_done(line)                              #  (a ValueError family member)
                self.assertEqual(e.last, {"generated": 1, "note": "previous leg"},
                                 "malformed MANDATORY core: self.last NOT half-replaced - " + why)

    def test_malformed_optional_group_reads_absent_never_raises(self):
        # core valid, ONE optional group corrupt: that group (and positional groups behind it) read
        # as absent - like an older engine line; the parent instead raised AFTER assigning the core
        # (a partial self.last AND a dead request). (tail, present-must-stay, must-be-absent)
        CASES = [("\u00b2 0 0", (), ("drafts_accepted", "drafts_offered", "reused")),
                 ("7 \u00b2 0", (), ("drafts_accepted", "drafts_offered", "reused")),
                 ("7 9 \u00b2", (), ("drafts_accepted", "drafts_offered", "reused")),
                 ("7 9 2 5 5 1 1 \u00d7.5 58 6", ("drafts_accepted", "hits", "lookups"),
                  ("ram_blobs", "file_blobs", "file_mb", "prompt_read", "offloaded"))]
        for tail, keep, absent in CASES:
            line = f"DONE {self.CORE} {tail}"
            with self.subTest(line=line):
                e = self.feed(line)                      # the parent RAISES here: red before fix
                self.assertEqual(e.last["generated"], 3, "the core landed")
                for k in keep:
                    self.assertIn(k, e.last, f"valid groups BEFORE the corrupt one stay: {k}")
                for k in absent:
                    self.assertNotIn(k, e.last, f"the corrupt group and everything positional behind it: {k}")

    def test_kv_fields_never_fabricate_and_never_raise(self):
        for kv in ("chunk=\u00b2", "chunk=\u0667", "chunk=+7", "chunk=07", "chunk=" + "9" * 100,
                   "experts_streamed=\u00b2", "chunk=-3", "chunk=", "chunk=1.5"):
            line = f"DONE {self.FULL15} {kv} experts_streamed=2"     # kv sits behind the 16 positional
            with self.subTest(line=line):                            #  words - parent reads it THERE
                e = self.feed(line)                      # must not raise (today's try/except too)
                self.assertNotIn("chunk", e.last, f"malformed kv value fabricates nothing: {kv}")
                self.assertEqual(e.last["experts_streamed"], 2, "the good kv sibling still lands")

    def test_boundary_ll_printable_stays_exact(self):
        LL = str(2 ** 63 - 1)
        e = self.feed(f"DONE {LL} {LL} 1.0 1.0 stop {LL} {LL} {LL}")
        self.assertEqual((e.last["generated"], e.last["prompt_tokens"], e.last["reused"]),
                         (2 ** 63 - 1,) * 3, "the printable %lld boundary")


class DoneCallerHarness(unittest.TestCase):
    """full legs through the control/solo readers: the engine's own DONE never fabricates figures
    into the request's timings/history, and an optional-group lie never kills the request."""

    def setUp(self):
        self.r = None

    def start(self, more):
        import serve.test_parallel as tp
        self.r = tp.ParallelService()
        self.r.start(0, more=tuple(more))   # batch=0: the solo reader AND engine.last readable
        self.e = self.r.engine

    def tearDown(self):
        if self.r:
            self.r.tearDown()

    def chat(self, text="Name two rivers in Germany"):
        import urllib.error
        try:
            return self.r.chat(text, 200)
        except urllib.error.HTTPError as ex:
            return ("http", ex.code, ex.read().decode(errors="replace")[:200])
        except Exception as ex:                                # noqa: BLE001
            return ("gone", type(ex).__name__)

    def figures(self):
        return dict(self.e.last or {})

    def history_tail(self):
        with self.r.svc.status_lock:
            return dict(self.r.svc.history[-1]) if self.r.svc.history else {}

    # ---- red on the parent BY BEHAVIOUR ---------------------------------------

    def test_arabic_generated_fabricates_no_figures(self):
        # parent: int('\u0667') == 7 PARSES -> the request's own figures say 7 tokens generated
        # (request_stats carries it into /status history's engine_generated and decode_tok_s).
        self.start(("--done-bad", "arabic"))
        res = self.chat()
        self.assertNotEqual(self.figures().get("generated"), 7,
                            "a wire figure the engine never produced never enters self.last")
        self.assertNotEqual(self.history_tail().get("engine_generated"), 7)

    def test_huge_counter_fabricates_no_figures(self):
        self.start(("--done-bad", "huge"))
        self.chat()
        gen = self.figures().get("generated")
        self.assertTrue(gen is None or gen < 10 ** 6, f"beyond-%lld counter never lands (saw {gen!r})")
        self.assertTrue((self.history_tail().get("engine_generated") or 0) < 10 ** 6)

    def test_nonfinite_ms_never_poisons_status_history(self):
        # parent: float() parses 'inf' and unicode digits -> history's prompt_ms/decode_ms go
        # non-finite (decode_tok_s/hit_rate math on them yields NaN in the /status JSON).
        self.start(("--done-bad", "infms"))
        self.chat()
        for key in ("prompt_ms", "decode_ms"):
            v = self.history_tail().get(key)
            self.assertTrue(v is None or (v == v and v != float("inf")),
                            f"history {key} must stay finite (saw {v!r})")

    def test_foreign_finish_word_never_lands_in_the_figures(self):
        # parent: any word was swallowed silently; the internal consumers (the #879 nonfinite retry,
        # the stop/length leg logic, the cancel accounting) then run on a word no emitter can print.
        self.start(("--done-bad", "finish"))
        self.chat()
        self.assertNotEqual(self.figures().get("finish"), "stopped",
                            "finish is the emitter's closed set - a foreign word fabricates nothing")

    def test_corrupt_optional_group_never_kills_the_request(self):
        # parent: the core dict was assigned, THEN the group's int() raised -> a PARTIAL self.last
        # and a 400 blamed on the client for the engine's own line.  Fixed: the group reads absent,
        # the request completes with the core's true figures.
        self.start(("--done-bad", "group9"))
        res = self.chat()
        self.assertNotIsInstance(res, tuple, "an optional-group lie may not end the request")
        f = self.figures()
        self.assertEqual(f.get("generated"), 10, "the true core figures landed")
        self.assertNotIn("drafts_accepted", f, "the corrupt group is absent as a whole, never half")

    def test_kv_arabic_chunk_fabricates_nothing(self):
        self.start(("--done-bad", "kvarabic"))
        res = self.chat()
        self.assertNotIsInstance(res, tuple)
        self.assertNotEqual(self.figures().get("chunk"), 7, "int('\\u0667') never enters the figures")

    # ---- r02 owner policy: a corrupt mandatory DONE is the ENGINE's fault ------------------

    def test_malformed_mandatory_core_ends_as_an_engine_fault(self):
        # OWNER decision 22.10 (was: the parent's bare raise landed in the 400 handler and blamed
        # the client for the engine's own completion line).  Where the READER reaches the DONE line
        # (the consumer is still pulling - no local stop/EOS break before it), the request fails
        # with 503 server_error, the message names the DONE line; figures stay clean (no half
        # update).  The harness fake's answer ends on its EOS token, so BOTH local break levers
        # (svc.stop_ids, server.EOS_IDS) are set empty to keep the reader pulling to the DONE.
        # RED on parent AND on r01-head on PURPOSE: the terminal class changed by owner order.
        # (The drain-side variant - answer already complete, DONE only syncs the wire - is the
        # separate, DELIBERATE degradation pinned right below.)
        from unittest import mock
        for mode in ("sup", "trunc"):
            with self.subTest(mode=mode):
                self.start(("--done-bad", mode))
                with mock.patch.object(self.r.svc, "stop_ids", set()):
                    res = self.chat()
                self.assertIsInstance(res, tuple, f"{mode}: the request must end as an error")
                self.assertEqual(res[0], "http")
                self.assertEqual(res[1], 503, f"{mode}: engine fault, never the 400 handler")
                self.assertIn("server_error", res[2])
                self.assertIn("DONE", res[2], "the message names the engine's end line")
                self.assertIsNone(self.figures().get("generated"),
                                  "no figure of the corrupt DONE ever entered self.last")

    def test_corrupt_done_seen_only_in_the_drain_degrades_to_missing_stats(self):
        # DELIBERATE r02 choice (documented for review): when the answer had ALREADY completed
        # locally (EOS/stop/max_new break) and the DONE only syncs the wire inside the close/drain,
        # a lying end line must NOT kill the finished answer (parent and r01: 400 killed a complete
        # reply over the engine's broken stats line).  The request completes with the SERVER-side
        # true counts; the wire figures read as absent - never invented, logged once.
        self.start(("--done-bad", "infms"))
        res = self.chat()
        self.assertNotIsInstance(res, tuple, "the finished answer is not killed by a lying stats line")
        self.assertEqual(res["usage"]["completion_tokens"], 10)
        f = self.figures()
        self.assertNotEqual(f.get("prompt_ms"), float("inf"), "no non-finite figure landed")
        with self.r.svc.status_lock:
            h = dict(self.r.svc.history[-1]) if self.r.svc.history else {}
        for key in ("prompt_ms", "decode_ms"):
            v = h.get(key)
            self.assertTrue(v is None or (v == v and v != float("inf")),
                            f"history {key} stays finite (saw {v!r})")

    def test_kv_superscript_inert_today_too(self):
        self.start(("--done-bad", "kvsup"))
        res = self.chat()
        self.assertNotIsInstance(res, tuple, "the kv try/except already swallowed junk")
        self.assertNotIn("chunk", self.figures())


class DoneCallerCleanup(unittest.TestCase):
    """r02 (both Codex counterexamples, permanent): a corrupt DONE ends the request as the engine
    fault WITHOUT breaking the lifecycle - no second wait for an already-consumed end, no held
    control lock, no drain that raises over the request's real fault.  Driven directly over the
    REAL generate()/_drain_control functions with scripted queues (batch harness cannot reach the
    solo reader), plus the full batch=2 service for the follow-up client."""

    # ---- solo reader: the counterexample "one bad record, then silence" ---------------------

    def solo_fixture(self, lines, silence_s=1):
        e = object.__new__(server.StrataEngine)
        e.alive, e.batch, e.gen = (lambda: True), 0, 0
        e.lines = queue.Queue()
        for ln in lines:
            e.lines.put(ln)

        class _Stdin:
            def __init__(self): self.text = ""
            def write(self, s): self.text += s
            def flush(self): pass
        e.proc = types.SimpleNamespace(stdin=_Stdin(), pid=0)
        e._send, e.can_stop, e.silence_s, e.last = (lambda line: None), False, silence_s, {}
        e.sampling_keys, e.exit_code = (lambda s: ""), (lambda: 0)
        e.progress, e.progress_ms, e.reused, e.prefill_tok_s_mean = None, 0, 0, None
        return e

    def test_corrupt_done_is_the_fault_not_a_drain_timeout(self):
        # r01-head: _parse_done raised BEFORE done=True, the finally STOP-drained onto the end the
        # wire had ALREADY delivered, and EngineSilent replaced the real fault after the silence
        # window.  Fixed: DONE marks the end before its figures are read -> the request fails with
        # StreamCorrupted, no drain, no wait.
        e = self.solo_fixture(["DONE 7 60 inf 1.0 stop 0 0 0"])
        with self.assertRaises(server.StreamCorrupted):
            list(e.generate([1, 2, 3], 4, {}, threading.Event()))
        self.assertIsNone(e.progress)                       # the reader never re-entered its loop

    def test_valid_done_end_still_clean_after_the_done_flag_move(self):
        # pin (green parent and r01 too): the reordering must not touch the normal end.
        e = self.solo_fixture(["T 5", "DONE 1 60 1.0 1.0 stop 0 0 0"])
        toks = list(e.generate([1, 2, 3], 4, {}, threading.Event()))
        self.assertEqual([t for t in toks if t is not None], [5])
        self.assertEqual(e.last["generated"], 1)

    # ---- drains: a lying end line must NEVER raise out of them ------------------------------

    def drain_fixture(self, lines):
        e = object.__new__(server.StrataEngine)
        e.alive, e.batch, e.gen = (lambda: True), 0, 0
        e.lines = queue.Queue()
        for ln in lines:
            e.lines.put(ln)
        e.last = {}
        return e

    def test_drain_control_never_raises_on_a_lying_done(self):
        # counterexample 2: the SECOND parse error sat in the finally drain; it escaped the
        # finally's `except EngineDied`, skipped ctl.release() (and admit/reserved bookkeeping)
        # and held the control lock for every later request.  The end line ends the drain
        # whatever its figures say; the lying leg's figures read as absent.
        bad = "DONE 7 60 inf 1.0 stop 0 0 0"
        e = self.drain_fixture([bad])
        self.assertEqual(e._drain_control("DONE", timeout=2), bad, "the end line is returned")
        self.assertEqual(e.last, {}, "a lying end leg fabricates no figures")
        e2 = self.drain_fixture([bad, "BADM 1 0"])
        self.assertEqual(e2._drain_control("BADM", timeout=2), "BADM 1 0",
                         "the drain walks over the corrupt DONE to its BADM - the finally reaches ctl.release()")
        self.assertEqual(e2.last, {})

    def test_solo_stop_drain_does_not_replace_the_outgoing_fault(self):
        # consumer stopped early -> STOP -> drain: a corrupt DONE there must not raise OVER the
        # original reason (parent: its bare int() could raise right here - pre-existing hole,
        # reachable with an 'x' record long before A03h).
        e = self.solo_fixture(["T 5", "DONE \u00b2 60 1.0 1.0 stop 0 0 0"])
        gen = e.generate([1, 2, 3], 4, {}, threading.Event())
        tok = next(gen, None)                                  # pull ONE token, then leave early:
        while tok is None:                                     #  the generator parks at its yield
            tok = next(gen, None)
        self.assertEqual(tok, 5)
        gen.close()                                   # early exit: the finally STOPs and drains
        # (the generator closes cleanly - the corrupt drain DONE is logged, figures stay absent)
        self.assertEqual(e.last, {}, "the drained lying end never entered self.last")

    # ---- full service batch=2: engine fault, free lock, working follow-up client ------------

    def setUp(self):
        self.r = None

    def tearDown(self):
        if self.r:
            self.r.tearDown()

    def post(self, text, stream=False):
        import json as _json
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": 10,
                "reasoning_effort": "none", "stream": stream}
        req = urllib.request.Request(self.r.base + "/v1/chat/completions", data=_json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return ("ok", resp.read().decode())
        except urllib.error.HTTPError as ex:
            return ("http", ex.code, ex.read().decode(errors="replace")[:300])
        except Exception as ex:                                  # noqa: BLE001
            return ("gone", type(ex).__name__)

    def test_corrupt_done_503_then_a_healthy_follow_up(self):
        # Owner policy at service level (engine fault -> 503, then a healthy follow-up): the reader
        # must be the one to reach the DONE, so the fake's EOS token must not break the consumer
        # loop (svc.stop_ids empty).  r01-head: the corrupt line reached in the close-drain raised
        # there and the whole finished request died 400.  (The batch=2 lifecycle machinery is
        # pinned at function level above - driving it with empty stop_ids asks the FAKE to continue
        # a solo leg as a slot it cannot run, a harness artifact, not a product state.)
        from unittest import mock
        import serve.test_parallel as tp
        self.r = tp.ParallelService()
        self.r.start(0, more=("--done-bad", "infms"))
        with mock.patch.object(self.r.svc, "stop_ids", set()):
            bad = self.post("Name two rivers in Germany")
        self.assertEqual(bad[0], "http")
        self.assertEqual(bad[1], 503, "engine fault before the answer")
        self.assertIn("server_error", bad[2])
        self.assertIsNone((self.r.engine.last or {}).get("generated"), "no figure landed")
        ok = self.post("Name one river in Germany")
        self.assertEqual(ok[0], "ok", "the follow-up client gets an answer: engine and queues healthy")
        data = json.loads(ok[1])
        self.assertEqual(data["usage"]["completion_tokens"], 10)

    def test_corrupt_done_after_stream_started_is_an_api_error_event(self):
        # owner policy half 2: the answer had already started -> API error event, not a dropped
        # connection, not a normal end (parent: inf PARSED - a quiet normal end: red on parent).
        from unittest import mock
        import serve.test_parallel as tp
        self.r = tp.ParallelService()
        self.r.start(0, more=("--done-bad", "infms"))
        with mock.patch.object(self.r.svc, "stop_ids", set()):   # the reader reaches the DONE itself
            res = self.post("Name two rivers in Germany", stream=True)
        self.assertEqual(res[0], "ok", "the stream closes cleanly")
        self.assertIn('"error"', res[1], "the API error event reports the engine fault mid-stream")
        self.assertNotIn('finish_reason": "stop', res[1], "a corrupt end never reports a normal stop")

    # ---- r03 (Codex single-DONE probe): the consumed end marker must reach the batch
    # ---- caller's finally BEFORE the figures can fail - one DONE then silence, no second
    # ---- STOP, no second drain, ctl released at once.  Real generate_batched/_control/finally,
    # _take_control replaced by the already-held lock, drain bounded to .05 s (observed min
    # get-wait .1 s) - no real 300 s wait, independent of stop_ids/fake-EOS.

    BAD = "DONE 7 60 inf 1.0 stop 0 0 0"

    def gb_fixture(self):
        e = object.__new__(server.StrataEngine)
        e.batch, e.alive, e.gen = 2, (lambda: True), 0
        e.sampling_keys, e.exit_code = (lambda s: ""), (lambda: 0)
        e.slot_cv, e.slot_busy, e.waiting = threading.Condition(), [False, False], 0
        e.ctl = threading.Lock()
        e.ctl.acquire()                                   # _take_control replaced: lock already held
        e.lines, e.last = queue.Queue(), {}
        sent = []
        e._send = sent.append

        def take(*a, **kw):                              # generator API: True, no yields
            if False:
                yield
            return True
        e._take_control = take
        drains = []
        orig = server.StrataEngine._drain_control

        def bounded(until, born=None):
            drains.append((until, e.lines.empty(), e.ctl.locked()))
            return orig(e, until, timeout=0.05, born=born)
        e._drain_control = bounded
        return e, sent, drains

    def test_single_consumed_done_never_redrains_the_solo_leg(self):
        e, sent, drains = self.gb_fixture()
        e.lines.put(self.BAD)                            # exactly ONE terminal record, then silence
        t = time.monotonic()
        with self.assertRaises(server.StreamCorrupted):  # the real fault still surfaces
            list(e.generate_batched([1, 2, 3], 4, {}, threading.Event()))
        self.assertEqual(drains, [], "no second drain onto an end the wire already delivered")
        self.assertNotIn("STOP", sent, "no second STOP after the consumed end")
        self.assertFalse(e.ctl.locked(), "ctl released: the finally ran to completion")
        self.assertLess(time.monotonic() - t, 5.0, "no 300 s hold on the control lock")
        self.assertTrue(e.ctl.acquire(blocking=False), "a follow-up call gets the control lock at once")

    def test_leg_that_never_saw_its_end_still_stops_and_drains(self):
        # NO blanket StreamCorrupted/parse-fault == finished admission: a leg that died BEFORE its
        # DONE (here the T-line parse, pre-existing bare int()) must still STOP and drain.
        e, sent, drains = self.gb_fixture()
        e.lines.put("T \u00b2")                          # dies before any end marker
        with self.assertRaises(ValueError):
            list(e.generate_batched([1, 2, 3], 4, {}, threading.Event()))
        self.assertEqual([d[0] for d in drains], ["DONE"], "the wire cleanup still runs")
        self.assertIn("STOP", sent, "STOP still follows a leg whose end is NOT consumed")

    def test_stale_end_stamp_never_gates_the_next_leg(self):
        # generation/leg fidelity: after one consumed DONE, the NEXT leg of the SAME engine that
        # dies before its end must still drain (per-leg reset of the stamp).
        e, sent, drains = self.gb_fixture()
        e.lines.put(self.BAD)
        with self.assertRaises(server.StreamCorrupted):
            list(e.generate_batched([1, 2, 3], 4, {}, threading.Event()))
        self.assertEqual(drains, [])                     # first leg: no drain (the fix)
        e.ctl.acquire()
        e.lines.put("T \u00b2")                          # second leg dies before its end
        with self.assertRaises(ValueError):
            list(e.generate_batched([1, 2, 3], 4, {}, threading.Event()))
        self.assertEqual([d[0] for d in drains], ["DONE"], "the stale stamp did not gate leg two")


class OptionalGroupAccounting(unittest.TestCase):
    """r02 documentation pin: reading a present-but-lying OPTIONAL group as absent is a DELIBERATE
    contract change (the parent raised there).  What falls out, exactly: reused/prompt_read/
    hits/lookups are absent (None), never invented - the carry merge skips, the /status fields go
    None, and prompt accounting (#471) falls back to the FULL prompt."""

    def setUp(self):
        self.r = None

    def tearDown(self):
        if self.r:
            self.r.tearDown()

    def test_lying_group_absent_and_accounting_falls_back(self):
        import serve.test_parallel as tp
        self.r = tp.ParallelService()
        self.r.start(0, more=("--done-bad", "group9"))        # drafts group starts with '\u00b2'
        res = self.r.chat("Name two rivers in Germany", 200)
        self.assertEqual(res["usage"]["completion_tokens"], 10, "the request completes normally")
        f = dict(self.r.engine.last or {})
        self.assertEqual(f.get("generated"), 10, "the valid core landed")
        for key in ("drafts_accepted", "drafts_offered", "reused", "prompt_read", "hits"):
            self.assertNotIn(key, f, f"the lying group and everything positional behind it: {key}")
        with self.r.svc.status_lock:
            h = dict(self.r.svc.history[-1])
        self.assertIsNone(h.get("reused"), "carry/status treat reused as unknown, never 0")
        self.assertEqual(h.get("prompt_tokens"), res["usage"]["prompt_tokens"],
                         "#471 fallback: prompt_read absent -> the FULL prompt counts, nothing invented")
        self.assertIsNone(h.get("hit_rate"), "hit rate unknown without lookups")


if __name__ == "__main__":
    unittest.main()
