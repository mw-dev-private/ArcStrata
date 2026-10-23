"""A disconnect reaches the real engine pipe during a quiet prompt read, before its first PP or token.

The native stand-in deliberately says nothing until STOP. This covers the boundary that a MockEngine watching
the cancel Event itself cannot test: HTTP disconnect -> Event -> STOP -> DONE/BADM -> the next request.
No GPU or model is used.
"""
import json
import io
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.request

from serve import server


FAKE = r'''import json, queue, sys, threading, time
from pathlib import Path
log = Path(sys.argv[sys.argv.index('--events') + 1])
batch = int(sys.argv[sys.argv.index('--batch') + 1])
commands, stop = queue.Queue(), threading.Event()
def note(event):
    with log.open('a') as f:
        f.write(json.dumps({'event': event, 'at': time.monotonic()}) + '\n')
def reader():
    for line in sys.stdin:
        line = line.strip()
        if line == 'STOP':
            note('stop')
            stop.set()
        else:
            commands.put(line)
    commands.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
if batch:
    print(f'INFO batch_slots={batch}', flush=True)
print('READY 4096 stop', flush=True)
count = 0
while True:
    line = commands.get()
    if line == 'QUIT':
        break
    if not line.startswith(('GEN ', 'BGEN ')):
        continue
    count += 1
    stop.clear()
    note('request_' + str(count))
    fields = line.split()
    slot = int(fields[1]) if fields[0] == 'BGEN' else None
    if count == 1:
        # The test's cleanup also sends STOP, so even a failed assertion leaves no long-lived child.
        stop.wait(15)
        time.sleep(.15)  # STOP was received, but the current chunk has not finished yet: it must drain.
        print('DONE 0 128 1 0 cancel 0 0 0', flush=True)
    else:
        print('T 90\nDONE 1 20 1 1 stop', flush=True)
    if slot is not None:
        print(f'BADM {slot} 0', flush=True)
'''


class QuietPromptCancel(unittest.TestCase):
    def check_hangup(self, api, stream, batch, max_new=1):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, events = root / 'engine.py', root / 'events.jsonl'
            script.write_text(FAKE, encoding='utf-8')
            real_popen = server.subprocess.Popen
            with mock.patch.object(server.subprocess, 'Popen', lambda cmd, **kw:
                                   real_popen([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = server.StrataEngine('strata', ['--batch', str(batch), '--events', str(events)])
            service = server.Service(engine, server.ByteTokenizer(),
                                     server.ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
            http = server.serve(service, port=0)
            port = http.server_address[1]
            sock = None
            reply, errors = [], []
            follower = None
            try:
                body = {'model': 'm', 'stream': stream, 'max_tokens': max_new,
                        'messages': [{'role': 'user', 'content': 'a long prompt'}],
                        'reasoning_effort': 'none'}
                if api == 'responses':
                    body = {'model': 'm', 'stream': stream, 'max_output_tokens': max_new,
                            'input': 'a long prompt', 'reasoning': {'effort': 'none'}}
                data = json.dumps(body).encode()
                path = {'openai': '/v1/chat/completions', 'anthropic': '/v1/messages',
                        'responses': '/v1/responses'}[api]
                sock = socket.create_connection(('127.0.0.1', port), timeout=5)
                sock.sendall(f'POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n'
                             f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)

                def has(event):
                    return events.exists() and any(json.loads(row)['event'] == event
                                                   for row in events.read_text().splitlines())

                deadline = time.monotonic() + 3
                while not has('request_1') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(has('request_1'), 'the first request did not reach the native pipe')
                closed = time.monotonic()
                sock.close()
                sock = None

                def follow():
                    data = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}],
                                       'max_tokens': 1, 'reasoning_effort': 'none'}).encode()
                    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=data,
                                                 headers={'Content-Type': 'application/json'})
                    try:
                        with urllib.request.urlopen(req, timeout=15) as response:
                            reply.append(json.load(response))
                    except Exception as exc:
                        errors.append(exc)

                follower = threading.Thread(target=follow)
                follower.start()
                deadline = closed + 2.5
                while not has('stop') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(has('stop'), 'disconnect must send STOP before a PP line or 10 s heartbeat')
                follower.join(timeout=2.5)
                self.assertFalse(follower.is_alive(), 'the cancelled request did not release the next request')
                self.assertEqual(errors, [])
                self.assertEqual(reply[0]['choices'][0]['message']['content'], 'Z')
                self.assertTrue(engine.alive(), 'cancellation must preserve the resident engine')
                self.assertEqual(reply[0]['usage']['completion_tokens'], 1)
                self.assertEqual(service.history[-1]['engine_generated'], 1,
                                 'the next request must record its own DONE, not the cancelled request')
                if batch:
                    self.assertFalse(any(engine.slot_busy), 'the cancelled admission must release its slot')
                else:
                    self.assertEqual(engine.last['finish'], 'stop', 'the next request must read its own DONE')
            finally:
                if sock is not None:
                    sock.close()
                engine._send('STOP')
                if follower is not None:
                    follower.join(timeout=3)
                http.shutdown()
                http.server_close()
                engine.close()

    def test_anthropic_stream(self):
        self.check_hangup('anthropic', True, 0)

    def test_batch_solo_stream(self):
        # A batch-enabled engine runs an isolated request with more than one output token through GEN first.
        self.check_hangup('anthropic', True, 2, max_new=2)

    def test_other_http_modes(self):
        for api in ('openai', 'anthropic', 'responses'):
            for stream in (False, True):
                for batch in (0, 2):
                    if (api, stream, batch) == ('anthropic', True, 0):
                        continue
                    with self.subTest(api=api, stream=stream, batch=batch):
                        self.check_hangup(api, stream, batch)


class QuietWaitPolicy(unittest.TestCase):
    def test_expired_silence_is_not_a_heartbeat(self):
        engine = object.__new__(server.StrataEngine)
        engine.proc = mock.Mock()
        engine.proc.poll.return_value = None
        engine.proc.stdin = io.StringIO()
        engine.lines = mock.Mock()
        engine.can_stop, engine.silence_s, engine.log_path = True, 10, None
        now = [0.0]

        def clock():
            now[0] += .001                 # time also passes between calculating a wait and starting it
            return now[0]

        def get(timeout):
            now[0] += timeout
            raise queue.Empty

        engine.lines.get.side_effect = get
        with mock.patch.object(server.time, 'monotonic', clock):
            gen = engine.generate([], 10, {}, threading.Event())
            with self.assertRaises(server.EngineSilent):
                next(gen)                  # must fail before yielding an overdue heartbeat to a slow consumer
        engine.proc.kill.assert_called_once()
        self.assertNotIn('STOP', engine.proc.stdin.getvalue())

    def test_polling_keeps_ten_second_heartbeats(self):
        for control in (False, True):
            with self.subTest(control=control):
                engine = object.__new__(server.StrataEngine)
                engine.proc = mock.Mock()
                engine.proc.poll.return_value = None
                engine.proc.stdin = io.StringIO()
                engine.lines = mock.Mock()
                engine.can_stop, engine.silence_s = True, 0
                engine._ctl_mode = 'solo'
                engine._send = mock.Mock()
                now = [0.0]
                waits = []

                def get(timeout):
                    waits.append(timeout)
                    if now[0] >= 21.0:
                        return 'DONE 0 1 1 0 length'
                    now[0] += timeout
                    raise queue.Empty

                engine.lines.get.side_effect = get
                cancel = threading.Event()
                with mock.patch.object(server.time, 'monotonic', lambda: now[0]):
                    gen = engine._control(cancel, lambda t: None) if control else engine.generate([1], 1, {}, cancel)
                    heartbeats = list(gen)
                self.assertEqual(heartbeats, [None, None], 'a quiet 21 seconds emits two heartbeats, not 42')
                self.assertTrue(all(0 < wait <= .5 for wait in waits), 'cancellation uses bounded blocking waits')
                engine._send.assert_not_called()
                self.assertNotIn('STOP', engine.proc.stdin.getvalue())


class SlotDecodeWaitPolicy(unittest.TestCase):
    """A03a: the batch slot decode loop (generate_batched after its admission) polls the slot queue with
    bounded blocking waits and keeps the quiet 10 s heartbeat separate - the solo path's pattern, now on
    the slot queue.  Virtual clock, fake queue: exact wait counts, heartbeat counts and BSTOP timing."""

    def fixture(self, now, waits, cancel, cancel_at=None, script_end=21.0):
        engine = object.__new__(server.StrataEngine)
        engine.proc = mock.Mock()
        engine.proc.poll.return_value = None
        engine.proc.stdin = io.StringIO()
        engine.alive = lambda: True
        engine.gen = 0
        engine.ctl = mock.Mock()
        engine.slot_cv = threading.Condition()
        engine.slot_busy = [True, False, False]      # "another request" holds slot 0: forces the slot path
        engine.slot_claims = {0}
        engine.slot_held, engine.slot_used, engine.slot_live = [[], [], []], [0.0, 0.0, 0.0], [None] * 3
        engine.waiting, engine.admitting = 0, 0
        engine.slot_releasing, engine._orphan_seen, engine.info = set(), {}, {}
        engine.pick_slot = lambda prompt: 1
        engine._may_go_solo = lambda *a: False
        engine.last = {}
        sent = []
        engine._send = lambda line: sent.append((line, now[0]))
        engine._take_control = lambda cancel_, n, **kw: self._returns_true()
        def control(cancel_, append, **kw):
            engine._ctl_result = (True, True)   # the real _control sets it too late to matter here, but the
            return []                           #  admission's reset right before the call must be undone
        engine._control = control
        engine._yielded = None

        class FakeQ:
            def empty(self):
                return True                    # the flush before admission finds nothing here

            def get(self, timeout):
                now[0] += timeout
                waits.append(timeout)
                if cancel_at is not None and now[0] >= cancel_at:
                    cancel.set()
                if now[0] >= script_end:
                    return "BDONE 1 1 stop 1.0"
                if waits and len(waits) == 1:
                    return "BT 1 65"                 # the slot's first token, then silence
                raise queue.Empty

        engine.slot_q = [None, FakeQ(), None]
        engine.slot_quarantine = set()             # A03b: the cleanup entry reads it; quiet-cancel ends land there
        return engine, sent

    @staticmethod
    def _returns_true():
        if False:
            yield
        return True

    def run_gen(self, engine, cancel):
        items = []
        with mock.patch.object(server.time, 'monotonic', lambda: self.now[0]):
            for item in engine.generate_batched([104, 105, 108], 5, {}, cancel):
                items.append(item)
        return items

    def test_slot_polling_keeps_ten_second_heartbeats(self):
        self.now = [0.0]
        waits = []
        cancel = threading.Event()
        engine, sent = self.fixture(self.now, waits, cancel)
        items = self.run_gen(engine, cancel)
        self.assertEqual(items, [65, None, None], 'a quiet slot keeps two 10 s heartbeats in 21 s, not dozens')
        self.assertTrue(all(0 < w <= .5 for w in waits), 'the slot decode polls with bounded blocking waits')
        self.assertEqual([s for s, _ in sent if 'BSTOP' in s], [], 'no BSTOP without a cancel')
        self.assertFalse(engine.slot_busy[1], 'the finished slot is released')

    def test_slot_cancel_during_silence_stops_at_its_own_wait(self):
        self.now = [0.0]
        waits = []
        cancel = threading.Event()
        engine, sent = self.fixture(self.now, waits, cancel, cancel_at=1.5)
        items = self.run_gen(engine, cancel)
        stops = [(t, at) for s, at in sent if 'BSTOP' in s for t in [s]]
        self.assertEqual(len(stops), 1, 'exactly one BSTOP for the cancelled slot')
        self.assertLessEqual(stops[0][1], 2.5,
                             'the cancel must reach BSTOP at the next bounded wait, not after the 10 s timeout')
        self.assertEqual(items.count(None), 2, 'polling does not add heartbeats (still 10 s each)')
        self.assertFalse(engine.slot_busy[1])

    # A03 review r02 (generation isolation, predates the polling change): a restart replaces self.slot_q and
    # the slot arrays (gen bumps in __init__). A consumer paused at a real yield must then never read, stop,
    # rewrite or free anything of the SUCCESSOR generation - its own pump's EOF None lands in the old queues.

    def swap_generation(self, engine, successor):
        engine.gen += 1                                # functional change of restart(), as __init__ does
        engine.slot_busy = [False, True, False]        # slot 1 now belongs to a NEW request
        engine.slot_claims = {1}
        engine.slot_held = [[], [999], []]
        engine.slot_live, engine.slot_used = [None] * 3, [0.0] * 3
        engine.slot_q = [None, successor, None]

    class SuccessorQ:
        def __init__(self):
            self.reads = []

        def get(self, timeout):
            self.reads.append(timeout)
            return "BDONE 1 1 stop 1.0"

    def test_generation_change_after_a_yield_never_touches_the_successor(self):
        self.now = [0.0]
        cancel = threading.Event()
        engine, sent = self.fixture(self.now, [], cancel)
        successor = self.SuccessorQ()
        with mock.patch.object(server.time, 'monotonic', lambda: self.now[0]):
            gen = engine.generate_batched([104, 105, 108], 5, {}, cancel)
            self.assertEqual(next(gen), 65)            # the consumer is paused at a real token (HTTP write)
            self.swap_generation(engine, successor)
            cancel.set()
            with self.assertRaises(server.EngineDied):
                list(gen)                              # the old request ends as engine death - ...
        self.assertEqual(successor.reads, [], 'the old consumer never reads the successor queue')
        self.assertNotIn('BSTOP 1', [s for s, _ in sent], 'no stop reaches generation 1')
        self.assertTrue(engine.slot_busy[1] and 1 in engine.slot_claims, 'the new request still owns its slot')
        self.assertEqual(engine.slot_held[1], [999], 'the successor cache is untouched')

    def test_generation_change_during_a_queue_wait_sends_no_stop(self):
        self.now = [0.0]
        waits = []
        cancel = threading.Event()
        engine, sent = self.fixture(self.now, waits, cancel)
        successor = self.SuccessorQ()

        def get(timeout):                              # the wait itself ends on a generation change
            self.now[0] += timeout
            waits.append(timeout)
            if len(waits) == 3:
                self.swap_generation(engine, successor)
                cancel.set()                           # a cancel racing the swap must not pick the successor
            if len(waits) == 1:
                return "BT 1 65"
            raise queue.Empty

        engine.slot_q[1].get = get
        with mock.patch.object(server.time, 'monotonic', lambda: self.now[0]):
            gen = engine.generate_batched([104, 105, 108], 5, {}, cancel)
            self.assertEqual(next(gen), 65)
            with self.assertRaises(server.EngineDied):
                list(gen)
        self.assertEqual(successor.reads, [], 'no read of the successor queue after the wait')
        self.assertNotIn('BSTOP 1', [s for s, _ in sent], 'no BSTOP for a generation this request died with')

    def test_leftover_line_racing_the_generation_change_is_not_consumed(self):
        self.now = [0.0]
        waits = []
        cancel = threading.Event()
        engine, sent = self.fixture(self.now, waits, cancel)
        successor = self.SuccessorQ()

        def get(timeout):                              # the OLD queue delivers a leftover line exactly as the
            self.now[0] += timeout                     #  generation flips: it must not be applied to anything
            waits.append(timeout)
            if len(waits) == 1:
                return "BT 1 65"
            self.swap_generation(engine, successor)
            return "BDONE 1 1 stop 1.0"

        engine.slot_q[1].get = get
        with mock.patch.object(server.time, 'monotonic', lambda: self.now[0]):
            gen = engine.generate_batched([104, 105, 108], 5, {}, cancel)
            self.assertEqual(next(gen), 65)
            with self.assertRaises(server.EngineDied):
                list(gen)
        self.assertEqual(engine.last, {}, 'the gone generation never applies a BDONE of the swap instant')
        self.assertEqual(engine.slot_held[1], [999], 'no held/cache write through the racing line')
        self.assertTrue(engine.slot_busy[1] and 1 in engine.slot_claims, 'the new request keeps its slot')

    def test_consumer_close_after_generation_change_leaves_the_new_slot_untouched(self):
        self.now = [0.0]
        cancel = threading.Event()
        engine, sent = self.fixture(self.now, [], cancel)
        successor = self.SuccessorQ()
        with mock.patch.object(server.time, 'monotonic', lambda: self.now[0]):
            gen = engine.generate_batched([104, 105, 108], 5, {}, cancel)
            self.assertEqual(next(gen), 65)
            self.swap_generation(engine, successor)
            gen.close()                                # the consumer leaves: its finally must not act either
        self.assertNotIn('BSTOP 1', [s for s, _ in sent], 'no stop, no background BDONE wait on generation 1')
        self.assertTrue(engine.slot_busy[1] and 1 in engine.slot_claims, 'the new request keeps busy and claim')
        self.assertEqual(engine.slot_held[1], [999], 'no held/cache rewrite through the gone generation')
        self.assertEqual(successor.reads, [], 'no queue reads at all after the swap')

    def test_resume_at_a_yield_never_offers_the_successor_slot_for_solo(self):
        # own r02 re-review (gate gap): _may_go_solo and its BSTOP run right AFTER `yield t`, before the loop
        # top - with the REAL _may_go_solo (fixture default stubs it off) a swapped generation would be read
        # (busy/waiting/info) and BSTOPped.  This keeps that path pinned.
        self.now = [0.0]
        cancel = threading.Event()
        engine, sent = self.fixture(self.now, [], cancel)
        del engine._may_go_solo                          # the real method: swap state = exactly one busy slot,
        engine.info = {"slot_cache": 1}                  #  nobody waiting, budget 64 > SOLO_AGAIN_MIN_LEFT
        successor = self.SuccessorQ()
        with mock.patch.object(server.time, 'monotonic', lambda: self.now[0]):
            gen = engine.generate_batched([104, 105, 108], 64, {}, cancel)
            self.assertEqual(next(gen), 65)              # paused at the token yield (HTTP write)
            self.swap_generation(engine, successor)
            with self.assertRaises(server.EngineDied):
                list(gen)
        self.assertNotIn('BSTOP 1', [s for s, _ in sent], 'no solo-handover stop may reach generation 1')
        self.assertEqual(successor.reads, [], 'the successor queue stays unread')
        self.assertTrue(engine.slot_busy[1] and 1 in engine.slot_claims, 'the new request keeps its slot')


class SlotDecodeCancel(unittest.TestCase):
    """A03a on the real adapter with a fake batch engine: a cancel set while the decode loop is inside
    its queue wait must produce BSTOP within the poll window; other slots stay untouched; a silent engine
    death still raises.  No HTTP, no GPU, slot 0 pre-marked busy (the fixture that forces slot decode)."""

    FAKE = r'''import json, queue, sys, threading, time
from pathlib import Path
args = sys.argv[1:]
log = Path(args[args.index("--events") + 1])
batch = int(args[args.index("--batch") + 1])
def note(ev, **kw):
    with log.open("a") as f:
        f.write(json.dumps({"event": ev, "at": round(time.monotonic(), 3), **kw}) + "\n")
cmds = queue.Queue()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l.startswith("BSTOP"):
            b = int(l.split()[1])
            note("BSTOP", slot=b)
            print("BDONE %d 1 cancel 1.0" % b, flush=True)
        else:
            cmds.put(l)
    cmds.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO batch_slots=%d" % batch, flush=True)
print("READY 4096 stop", flush=True)
while True:
    line = cmds.get()
    if line is None or line == "QUIT":
        break
    if line.startswith("BGEN "):
        f = line.split()
        slot, max_new = int(f[1]), int(f[2])
        ids = [int(x) for x in f[-1].split(",")]
        pb = bytes(x & 255 for x in ids)
        note("BGEN", slot=slot, fast=b"FAST" in pb)
        print("DONE 0 %d 1.0 0.0 stop 0 0 0" % len(ids), flush=True)
        print("BADM %d 1" % slot, flush=True)
        if b"DIE" in pb:
            print("BT %d 65" % slot, flush=True)
            sys.exit(0)                                  # the engine dies mid-silence
        if b"FAST" in pb:
            for k in range(min(max_new, 3)):
                time.sleep(0.05)
                print("BT %d %d" % (slot, ord("a") + k), flush=True)
            print("BT %d 257" % slot, flush=True)
            print("BDONE %d 4 stop 1.0" % slot, flush=True)
        else:
            print("BT %d 65" % slot, flush=True)         # first token, then dead silence until BSTOP
            note("silent", slot=slot)
'''

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="slot-cancel-")
        script, self.events = Path(self.directory) / "engine.py", Path(self.directory) / "events.jsonl"
        script.write_text(self.FAKE, encoding="utf-8")
        real_popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real_popen([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = server.StrataEngine("strata", ["--batch", "3", "--events", str(self.events)])
        with self.engine.slot_cv:                        # fixture: "another request" holds slot 0
            self.engine.slot_busy[0] = True
            self.engine.slot_claims.add(0)

    def tearDown(self):
        self.engine.close()

    def drive(self, prompt, max_new, cancel, sink):
        def run():
            try:
                for item in self.engine.generate_batched(prompt, max_new, {}, cancel):
                    sink.append((item, time.monotonic()))
            except Exception as exc:
                sink.append((type(exc).__name__, time.monotonic()))
            sink.append(("END", time.monotonic()))
        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th

    def wait_for(self, needle, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.events.exists() and needle in self.events.read_text():
                return
            time.sleep(0.01)
        self.fail("the fake engine never logged: " + needle)

    def rows(self, name):
        return [json.loads(r) for r in self.events.read_text().splitlines() if '"' + name + '"' in r]

    def test_cancel_in_a_silent_slot_queue_reacts_within_the_poll_window(self):
        cancel, sink = threading.Event(), []
        th = self.drive([ord(c) for c in "SLOWX"], 5, cancel, sink)
        self.wait_for('"event": "silent"')
        time.sleep(0.8)                                  # the loop is certainly inside its get()
        t0 = time.monotonic()
        cancel.set()
        th.join(timeout=15)
        self.assertFalse(th.is_alive(), 'the cancelled decode must finish, not hang')
        elapsed = sink[-1][1] - t0
        self.assertLessEqual(elapsed, 1.5,
                             f'cancel to end took {elapsed:.2f} s: BSTOP must not wait for the 10 s queue '
                             f'timeout (tolerance: one poll of 0.5 s plus scheduling and the BDONE roundtrip)')
        stops = self.rows("BSTOP")
        self.assertEqual(len(stops), 1, 'exactly one BSTOP')
        slot = next(r["slot"] for r in self.rows("BGEN") if not r["fast"])
        self.assertEqual(stops[0]["slot"], slot, 'BSTOP addresses the cancelled slot')
        self.assertFalse(self.engine.slot_busy[slot], 'the cancelled slot is released after its BDONE')
        self.assertTrue(self.engine.alive(), 'a slot cancel keeps the resident engine')

    def test_two_slots_only_the_cancelled_one_stops(self):
        cf, sf = threading.Event(), []
        ck, sk = threading.Event(), []
        th_f = self.drive([ord(c) for c in "FASTX"], 5, cf, sf)
        time.sleep(0.2)
        th_s = self.drive([ord(c) for c in "SLOWX"], 5, ck, sk)
        self.wait_for('"event": "silent"')
        time.sleep(0.8)
        ck.set()
        th_s.join(timeout=15)
        th_f.join(timeout=15)
        self.assertEqual(len(self.rows("BSTOP")), 1, 'only the cancelled request gets a BSTOP')
        self.assertEqual([i for i, _ in sf if isinstance(i, int)], [97, 98, 99, 257],
                         'the other slot runs untouched to its own end')
        self.assertEqual([i for i, b in enumerate(self.engine.slot_busy) if b], [0],
                         'both real slots are released (only the fixture slot remains)')

    def test_engine_death_during_the_polling_still_raises(self):
        cancel, sink = threading.Event(), []
        th = self.drive([ord(c) for c in "DIEXX"], 5, cancel, sink)
        th.join(timeout=15)
        self.assertEqual(sink[-1][0], "END")
        self.assertEqual(sink[-2][0], "EngineDied", 'the None sentinel stays Engine-Ende, not a cancel wakeup')


if __name__ == '__main__':
    unittest.main()
