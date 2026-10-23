"""A03g - PP and YIELDED lines only say what the NATIVE emitter can actually say.

Contract (fixed BEFORE the tests; decisions marked): the emitter prints
`PP <position> <prompt tokens> <ms> <fresh tok/s>` - four non-negative fields, the first two as
%lld, the last two as %.0f/%.1f of a clock difference and a guarded quotient (generate.cpp:9297/
12511/12572/13422, SYCL identical) - and `YIELDED <slot> <tokens>` ONLY paired with a real give-way
(yielded_at/yielded_slot are set as a pair, emitted only for yielded_at >= 0; the initial -1 never
reaches the wire, generate.cpp:12751-52/12842-43/14619).  DECISION (sources, not assumption): a
NEGATIVE slot word in YIELDED is not a legitimate solo sentinel - it has no emitter source.

A PP line is a HEARTBEAT whatever its fields say (it ends silence before parsing).  A corrupt
NUMERIC word fabricates nothing: counters go through _num_field (never raises; '²' and '٧' rejected),
ms/rate through a safe float gate (ASCII, finite, >= 0 - today float() PARSES 'inf'/'nan', and an
infinite progress_ms later dies in prompt_progress' int() (SSE path) or lands as literal 'Infinity'
in /status JSON).  An invalid ms/rate word means THAT FIELD is absent (0.0 / None - the same reading
as today's short line); the counters of a valid prefix still count.  The 3-field historical and the
5-field forward shape stay exactly as today.
YIELDED: slot and count through _num_field; the HAND-BACK binds at consumption - a yield claim only
counts for this request's own slot/reserved, and its count cannot exceed what the server sent as
this prompt (native q <= ids.size()).  An ignored YIELDED line takes the existing NO-YIELD path
(the prompt is re-read without a held prefix) - no new lifecycle policy, and progress junk never
ends or aborts a request (it is optional output; a missing YIELDED is already a handled case).
r02 (after Codex's independent probe): the COUNTER words are additionally bounded to printf %lld's
own printable range ([0, 2**63-1] - grammar from the emitter, the A03f %d precedent, NOT an
arbitrary cap): a canonical but unprintable word like 10**1000 fed the solo #481 allowance math
and RAISED OverflowError in the live generator (r01 and parent).  And the derived allowance may
only be REPLACED by a FINITE result: rate 5e-324 is finite and > 0 yet 3.0 * 30 / it == inf, which
disarmed the silence deadline entirely - a non-finite computed value reads as that field absent
(the same treatment as today's missing 5th field); the formula, the timeout policy and every
valid figure stay untouched.  Boundary of the guard, recorded honestly (review nit): it filters
NON-FINITE results only - a huge but FINITE allow (rate 1e-300 -> ~1e20 s) stays parent behaviour
by design ("same formula, no caps"); the emitter's %.1f never prints such a rate anyway.

Fixtures: FAKE_BATCH opt-in `--pp-bad sup|arabic|text|inf` (one corrupt PP per run; arabic hits the
read's LAST PP; inf sleeps 0.6 s so a probe can observe the poisoned field) and
`--yielded-bad sup|own|foreign` (a YIELDED line with no BYIELD behind it).  Defaults byte-identical.
FRAGILITY (recorded): the harness rows need at least one PP line per read - the fake emits PP every
CH=32 tokens, so the tokenised prompt must be >= 34 tokens (the current template + prompts: 62+; a
shrinking template would make these rows green-vacuous - the unit classes above do not depend on it).
The inf row's /status observation and the round-count pins use short bounded windows (fake sleep
0.6 s; fresh baseline service per count), never minute-long sweeps.
Baseline control: against the unfixed parent, the ²/'x' rows are HTTP 400 (a client blamed for the
engine's line), the arabic row fabricates the progress counter, and inf poisons progress_ms.
A forged YIELDED (own or foreign slot, bogus count) is CONTAINED by the parent's existing slot
binding - consequence proven inverted, pinned green on both trees; only its ² spelling goes red (400).
The negative-slot and valid/historical rows are pins green on both sides.
"""
import queue
import threading
import types
import time
import unittest
import urllib.error
from unittest import mock

import serve.test_parallel as tp
from serve import server


class ControlParse(unittest.TestCase):
    """real _control over a scripted control stream: parsing never raises, never fabricates."""

    def drive(self, lines):
        e = self.e = object.__new__(server.StrataEngine)
        e.lines, e.gen = queue.Queue(), 0
        for ln in lines:
            e.lines.put(ln)
        e._ctl_mode, e._ctl_result, e._ctl_erred = "solo", None, False
        e._last_done, e._yielded = None, None
        e.progress, e.progress_ms, e.prefill_tok_s_mean, e.reused = None, 0, None, 0
        e._send = lambda line: None
        e.sent = []
        for _ in e._control(threading.Event(), lambda t: None):   # runs to the DONE line
            pass
        return e

    def test_valid_forms_stay_exact(self):
        e = self.drive(["PP 30 60 1200 25.5", "DONE 0 60 5.0 0.0 stop 0 0 0"])
        self.assertEqual((e.progress, e.progress_ms, e.prefill_tok_s_mean),
                         ((30, 60), 1200.0, 25.5), "the active 4-field form (both trees)")
        e = self.drive(["PP 30 60", "DONE 0 60 5.0 0.0 stop 0 0 0"])
        self.assertEqual((e.progress, e.progress_ms, e.prefill_tok_s_mean),
                         ((30, 60), 0.0, None), "the historical 3-field form: stats absent, not guessed")
        e = self.drive(["PP 30 60 1200 25.5 9", "DONE 0 60 5.0 0.0 stop 0 0 0"])
        self.assertEqual((e.progress, e.progress_ms, e.prefill_tok_s_mean),
                         ((30, 60), 1200.0, 25.5), "the forward 5-field shape stays read as today")

    def test_invalid_counter_line_keeps_the_last_valid_figures(self):
        # today's semantics, preserved deliberately: a line whose COUNTER pair is malformed updates
        # NOTHING (the stale-but-true figures stand) - the whole update is gated on the pair, exactly
        # as the parent gated it on isdigit.  Mutation control for "gate the assignments separately".
        e = self.drive(["PP 30 60 1200 25.5", "PP \u00b2 60 9 9", "PP 40 60 inf inf",
                        "DONE 0 60 5.0 0.0 stop 0 0 0"])
        self.assertEqual(e.progress, (40, 60), "the second corrupt line still advanced (valid pair)")
        self.assertEqual(e.progress_ms, 0.0, "its corrupt ms word read as absent (0.0), not a poison")
        self.assertIsNone(e.prefill_tok_s_mean, "and its corrupt rate as absent (None)")

    def test_corrupt_words_never_raise_never_fabricate(self):
        # (line, expected progress, expected progress_ms, expected rate) - None progress: no update.
        LL = 2 ** 63 - 1                                 # %lld's own range: printf CANNOT print outside
        CASES = [                                        #  it (A03f %d precedent) - grammar, not a cap
            ("PP \u00b2 60 1 1", None, 0.0, None),          # int() raises TODAY: 400 blamed on client
            ("PP 30 \u00b2 1 1", None, 0.0, None),
            ("PP \u0667 60 1 1", None, 0.0, None),          # int() PARSES (7) TODAY: fabricated counter
            ("PP -5 60 1 1", None, 0.0, None),
            ("PP 30 -60 1 1", None, 0.0, None),
            ("PP 030 60 1 1", None, 0.0, None),             # %lld never prints a leading zero
            ("PP 30 60 x x", (30, 60), 0.0, None),          # float() raises TODAY: 400
            ("PP 30 60 inf inf", (30, 60), 0.0, None),      # float() PARSES today: poisons progress_ms
            ("PP 30 60 nan 1", (30, 60), 0.0, 1.0),
            ("PP 30 60 1 -2", (30, 60), 1.0, None),
            ("PP 30 60 -inf 1", (30, 60), 0.0, 1.0),
            ("PP 30 60 \u0667 1", (30, 60), 0.0, 1.0),      # float() reads Arabic-Indic digits TODAY
            ("PP 30 60 1e5 1_0", (30, 60), 100000.0, 10.0),   # exponent/underscore: accepted as
            ("PP  30  60 1 1", (30, 60), 1.0, 1.0),          #  numerically true (r03 decision);
                                                             #  split() collapses doubled spaces
            ("PP " + "9" * 1000 + " 60 1 1", None, 0.0, None),      # r02: _num_field converts it,
            (f"PP {LL + 1} 60 1 1", None, 0.0, None),               #  %lld has no source for it
            (f"PP 30 {LL + 1} 1 1", None, 0.0, None),
            (f"PP {LL} {LL} 1 1", (LL, LL), 1.0, 1.0),             # the printable boundary stays exact
        ]
        for line, prog, ms, rate in CASES:
            with self.subTest(line=line):
                e = self.drive([line, "DONE 0 60 5.0 0.0 stop 0 0 0"])   # must not raise, ever
                self.assertEqual(e.progress, prog, line)
                self.assertEqual(e.progress_ms, ms, line)
                self.assertEqual(e.prefill_tok_s_mean, rate, line)

    def test_yielded_line_forms(self):
        e = self.drive(["YIELDED 1 30", "DONE 0 60 5.0 0.0 cancel 0 0 0"])
        self.assertEqual(e._yielded, (1, 30), "the native paired form")
        for line in ("YIELDED -1 30",               # NO emitter source for a negative slot (see module
                     "YIELDED \u00b2 30",           #  docstring); int() raises TODAY: 400
                     "YIELDED \u0667 30", "YIELDED 01 30",
                     "YIELDED 1 -5", "YIELDED 1 07"):
            with self.subTest(line=line):
                e = self.drive([line, "DONE 0 60 5.0 0.0 cancel 0 0 0"])
                self.assertIsNone(e._yielded, f"not emitter-grammar: {line!r}")


class SoloGenerateParse(unittest.TestCase):
    """the batch=0 generate() reader carries the same PP parse as _control (one shared defect class) -
    driven directly over a scripted control stream, since the batch harness never reaches it."""

    def fixture(self, lines, silence_s=0):
        e = object.__new__(server.StrataEngine)
        e.alive, e.batch, e.gen = (lambda: True), 0, 0
        e.lines = queue.Queue()
        for ln in lines:
            e.lines.put(ln)

        class _Stdin:                                       # the GEN head goes here; nothing else does
            def __init__(self): self.text = ""
            def write(self, s): self.text += s
            def flush(self): pass
        e.proc = types.SimpleNamespace(stdin=_Stdin(), pid=0)
        e._send, e.can_stop, e.silence_s, e.last = (lambda line: None), False, silence_s, {}
        e.sampling_keys, e.exit_code = (lambda s: ""), (lambda: 0)
        e.progress, e.progress_ms, e.reused, e.prefill_tok_s_mean = None, 0, 0, None
        return e

    def drive(self, lines, silence_s=0):
        e = self.fixture(lines, silence_s)
        toks = list(e.generate([1, 2, 3], 4, {}, threading.Event()))
        return e, toks

    def test_corrupt_pp_words_never_raise_in_the_solo_reader(self):
        # parent: float('x') / int('\u00b2') inside the solo stream -> ValueError out of the generator;
        # float('inf') PARSES -> progress_ms goes non-finite, and prompt_progress' int() then kills the
        # SSE path with an unmapped OverflowError (that surface is batch=0-only: prompt_progress None
        # for batched engines - which is why the batch harness cannot see this half of the defect).
        e, toks = self.drive(["PP 30 60 x x", "PP \u00b2 60 1 1", "PP 30 60 inf inf",
                              "T 5", "DONE 1 60 1.0 1.0 stop 0 0 0"])
        self.assertEqual([t for t in toks if t is not None], [5], "the request ran to its end")
        self.assertEqual(toks.count(None), 3,
                         "even corrupt PP lines are HEARTBEATS (silence ends before parsing)")
        self.assertEqual(e.progress, (30, 60))
        self.assertEqual(e.progress_ms, 0.0, "the corrupt ms/rate words read as absent")
        self.assertIsNone(e.prefill_tok_s_mean)

    def test_solo_reader_valid_pp_line_passes_through_untouched(self):
        # what this fixture actually pins (the allowance FORMULA itself is local state, silence_s=0
        # here, and not observable without timing games): a VALID 4-field PP line parses identically
        # in the solo reader and the request runs to its end.
        e, toks = self.drive(["PP 30 60 1200 25.5", "T 5", "DONE 1 60 1.0 1.0 stop 0 0 0"])
        self.assertEqual((e.progress, e.progress_ms, e.prefill_tok_s_mean), ((30, 60), 1200.0, 25.5))
        self.assertEqual([t for t in toks if t is not None], [5])

    # ---- r02: the allowance ARITHMETIC with an ACTIVE silence deadline (silence_s>0) - the ----
    # ---- surface Codex's independent probe caught on r01 AND on the parent (pre-existing).     ----

    def test_huge_counter_never_crashes_the_solo_allowance_math(self):
        # r02 (Codex counterexample): a canonical 1000-digit position word converts fine via
        # _num_field, but PP_SLACK * chunk / rate then raises OverflowError (int too large to
        # convert to float) INSIDE the live generator - r01 and the parent both crash here.
        # %lld has no printable source for it: the pair reads as malformed (no update), and the
        # request runs on untouched.
        e, toks = self.drive([f"PP {'9' * 1000} 60 1 1", "T 5", "DONE 1 60 1.0 1.0 stop 0 0 0"],
                             silence_s=1)
        self.assertEqual([t for t in toks if t is not None], [5], "no arithmetic exception, request intact")
        self.assertIsNone(e.progress, "outside %lld range: the counter pair simply did not happen")

    def test_tiny_positive_rate_never_disarms_the_silence_deadline(self):
        # r02: rate 5e-324 is finite and > 0 (passes _float_field), but 3.0 * 30 / 5e-324 == inf:
        # allow goes inf and the #481 silence deadline NEVER fires again (r01 and parent both do).
        # Fixed: a non-finite computed allowance never replaces the finite deadline (the field reads
        # as absent, like today's missing 5th field) - with silence_s=1 EngineSilent must still fire.
        # No DONE is queued on purpose; the deadline is what ends this request. Bounded window, never
        # a minute-long sweep; red runs are released by a late DONE.
        e = self.fixture(["PP 30 60 1 5e-324"], silence_s=1)
        err: list = []

        def run():
            try:
                for _ in e.generate([1, 2, 3], 4, {}, threading.Event()):
                    pass
            except BaseException as ex:                          # noqa: BLE001
                err.append(ex)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(15)
        if t.is_alive():                                         # red path: release the blocked reader
            e.lines.put("DONE 0 60 1.0 1.0 stop 0 0 0")
            t.join(20)
        self.assertTrue(any(isinstance(x, server.EngineSilent) for x in err),
                        f"the finite silence deadline must survive a tiny rate (saw {err!r})")

    def test_deadline_and_cancel_unchanged_for_valid_figures(self):
        # pins green on r01, the parent AND r02: a normal rate keeps the request alive to its DONE,
        # and a cancel during silence ends the reader - honestly BOTH ways round (review nit): with
        # the engine's DONE still arriving, the finally-drain finds it and the return is clean; with
        # no DONE the timed drain rule (#481, "never an untimed wait here") ends it via EngineSilent
        # instead of hanging.  Identical on all three trees.
        e, toks = self.drive(["PP 30 60 1200 25.5", "T 5", "DONE 1 60 1.0 1.0 stop 0 0 0"], silence_s=1)
        self.assertEqual([t for t in toks if t is not None], [5])

        def run_cancel(e: "server.StrataEngine", cancel: threading.Event, out: list, err: list):
            try:
                out.extend(t for t in e.generate([1, 2, 3], 4, {}, cancel))
            except BaseException as ex:                          # noqa: BLE001
                err.append(ex)

        e2 = self.fixture([], silence_s=2)
        cancel = threading.Event()
        out, err = [], []
        t = threading.Thread(target=run_cancel, args=(e2, cancel, out, err), daemon=True)
        t.start()
        time.sleep(0.3)
        cancel.set()
        e2.lines.put("DONE 0 60 1.0 1.0 stop 0 0 0")             # the engine ends: the drain finishes
        t.join(15)
        self.assertFalse(t.is_alive(), "cancel during silence ends the reader")
        self.assertEqual(err, [], "clean return: the finally-drain found the DONE")

        e3, err3 = self.fixture([], silence_s=1), []
        t3 = threading.Thread(target=run_cancel, args=(e3, cancel, [], err3), daemon=True)
        t3.start()
        time.sleep(0.3)
        cancel.set()                                             # no DONE ever arrives
        t3.join(15)
        self.assertFalse(t3.is_alive())
        self.assertTrue(any(isinstance(x, server.EngineSilent) for x in err3),
                        f"no DONE: the timed drain ends with EngineSilent, never an endless wait (saw {err3!r})")


class ProgressHarness(unittest.TestCase):
    """full real legs over HTTP: progress junk is never a client 400, never a fabricated figure,
    and a YIELDED the engine never meant never rewrites the admission loop."""

    def setUp(self):
        self.r = None

    def start(self, more):
        self.r = tp.ParallelService()
        self.r.start(2, slot_cache=True, reuse=True, more=tuple(more))
        self.e = self.r.engine

    def tearDown(self):
        if self.r:
            self.r.tearDown()

    def chat(self, text="Name two rivers in Germany"):
        try:
            return self.r.chat(text, 200)
        except urllib.error.HTTPError as ex:
            return ("http", ex.code, ex.read().decode(errors="replace"))
        except Exception as ex:                                # noqa: BLE001
            return ("gone", type(ex).__name__)

    def two_chats(self, r=None):
        # NOTE (observed, pre-existing workload behaviour): with slot_cache this concurrent pair runs
        # FOUR admission rounds even with no flags at all (GEN, BGEN, BGEN, GEN - no YIELD lines) -
        # so the tests below compare COUNTS BETWEEN a flagged and an unflagged run, never absolutes.
        r = r or self.r
        res = {}
        chat = lambda k, t: res.setdefault(k, r.chat(t, 200))
        with mock.patch.object(server, "EOS_IDS", {257}):
            ta = threading.Thread(target=lambda: chat("a", "List three primes please"))
            tb = threading.Thread(target=lambda: chat("b", "Name two rivers in Germany"))
            ta.start(); tb.start(); ta.join(40); tb.join(40)
        return res

    def pair_rounds(self, more=()):
        probe = tp.ParallelService()
        probe.start(2, slot_cache=True, reuse=True, more=tuple(more))
        try:
            self.two_chats(probe)
            return len([l for l in probe.log.read_text().splitlines() if l.startswith("BGEN")])
        finally:
            probe.tearDown()

    def bgen_rounds(self):
        return [l for l in self.r.log.read_text().splitlines() if l.startswith("BGEN")]

    # ---- red on the parent BY BEHAVIOUR --------------------------------------

    def test_superscript_pp_word_is_never_a_client_400(self):
        # parent: int('\u00b2') raises inside the control read -> the 400 handler blames the client.
        self.start(("--pp-bad", "sup"))
        res = self.chat()
        self.assertNotIsInstance(res, tuple, "a corrupt PROGRESS line may never end the request")
        self.assertTrue(res["choices"][0]["message"]["content"].strip(), "full answer")

    def test_text_ms_pp_word_is_never_a_client_400(self):
        self.start(("--pp-bad", "text"))
        res = self.chat()
        self.assertNotIsInstance(res, tuple, "float('x') inside the read: same class, same verdict")
        self.assertTrue(res["choices"][0]["message"]["content"].strip())

    def test_arabic_pp_digit_fabricates_no_progress(self):
        # parent: int('\u0667') == 7 PARSES -> the read's final progress becomes (7, total): a figure
        # the engine never reached (this prompt's only chunk ends at 32); /status and the SSE progress
        # carry it.  Fixed: the line is skipped - this prompt then simply has no PP progress at all.
        self.start(("--pp-bad", "arabic"))
        res = self.chat()
        self.assertNotIsInstance(res, tuple)
        self.assertNotEqual((self.e.progress or (0, 0))[0], 7,
                            "no fabricated progress from a wire digit (fix: the corrupt line is skipped)")

    def test_infinite_ms_poisons_no_progress_field(self):
        # parent: float('inf') parses -> progress_ms = inf: prompt_progress' int() (SSE) dies with an
        # OverflowError mid-stream and /status would carry the literal 'Infinity'.
        self.start(("--pp-bad", "inf"))
        t = threading.Thread(target=self.chat)
        t.start()
        time.sleep(0.3)                                        # the fake sleeps 0.6 s after the line
        ms = self.e.progress_ms
        t.join(20)
        self.assertTrue(ms == ms and ms != float("inf") and ms >= 0.0,
                        f"progress_ms must never go non-finite (saw {ms!r})")
        self.assertNotEqual(self.e.prefill_tok_s_mean, float("inf"), "same for the rate")

    def test_superscript_yielded_word_is_never_a_client_400(self):
        self.start(("--yielded-bad", "sup"))
        res = self.two_chats()
        for k in ("a", "b"):
            self.assertIn(k, res, "both legs answered (no hang)")
            self.assertNotIsInstance(res[k], tuple, f"leg {k}: YIELDED junk is no client error")

    def test_forged_own_slot_yielded_leaves_no_state_harm(self):
        # Consequence PROVEN INVERTED (assignment: "Folgen zunächst nachweisen"): the parent loop's
        # existing slot binding (the continue branch compares the wire slot to THIS request's slot)
        # plus the re-admission's held-overwrite contain an own-slot line with a bogus count - no
        # extra admission round, no leftover state, answer intact. Pinned on both trees so a future
        # refactor of the admission loop cannot silently lose that containment.
        base = self.pair_rounds()
        self.start(("--yielded-bad", "own"))
        res = self.two_chats()
        for k in ("a", "b"):
            self.assertIn(k, res)
            self.assertNotIsInstance(res[k], tuple, f"leg {k}")
        flagged = len(self.bgen_rounds())
        self.assertEqual(flagged, base, f"rounds: flagged {flagged} vs baseline {base}")
        with self.e.slot_cv:
            self.assertFalse(any(self.e.slot_busy), "all slots idle")
            self.assertEqual(sorted(self.e.slot_claims), [], "no claim leaks")
            self.assertEqual(sorted(self.e.slot_releasing), [], "no watcher dangles")

    # ---- pins: green before AND after -----------------------------------------

    def test_foreign_slot_yielded_line_is_inert(self):
        # observed parent behaviour, kept: a line naming a slot this request does not own never
        # rewrites its admission loop (the binding above). Pinned so the fix may not change it.
        base = self.pair_rounds()
        self.start(("--yielded-bad", "foreign"))
        res = self.two_chats()
        for k in ("a", "b"):
            self.assertIn(k, res)
            self.assertNotIsInstance(res[k], tuple, f"leg {k}")
        self.assertEqual(len(self.bgen_rounds()), base, "no extra admission round")


if __name__ == "__main__":
    unittest.main()
