"""serve/test_mcp_transport.py - the MCP transports against scripted private servers (A10a): which answer belongs to which request,
what happens to a waiting call when the server ends, stalls or answers nonsense, and the stop button.  CPU only: child processes of
serve/mcp_scripted_server.py (stdio) and a loopback HTTP fixture; no real MCP server.

Tests named ..._POLICY pin behaviour that is an open question; they exist so a change is noticed, not because it is endorsed."""
import copy
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from serve import frontend, mcp, server

SCRIPTED = str(Path(__file__).with_name("mcp_scripted_server.py"))
SET = {**mcp.DEFAULTS, "start_timeout_s": 10.0, "timeout_s": 2.0}


def stdio_server(mode, timeout=2.0, start=10.0, env=None):
    log = tempfile.mktemp(suffix=".jsonl")
    s = mcp.McpServer("s", {"command": sys.executable, "args": ["-B", SCRIPTED, mode, log], "env": env or {}},
                      {**SET, "timeout_s": timeout, "start_timeout_s": start})
    return s, log


def calls_in(log):
    if not os.path.exists(log):
        return []
    with open(log) as f:
        return [json.loads(x) for x in f]


def timed(fn, *a):
    t0 = time.monotonic()
    try:
        out = ("result", mcp.result_text(fn(*a)))
    except mcp.McpError as e:
        out = (type(e).__name__, str(e))
    return out, time.monotonic() - t0


class StdioCase(unittest.TestCase):
    servers = ()

    def stdio(self, mode, **kw):
        s, log = stdio_server(mode, **kw)
        self.addCleanup(s.close)
        self.assertTrue(s.start(), s.error)
        return s, log

    def call(self, s, text="x", timeout=2.0, cancel=None, **extra):
        return timed(s.call, "t", {"text": text, **extra}, timeout, cancel)


class Matching(StdioCase):
    def test_two_concurrent_calls_answered_in_reverse_order(self):
        s, _ = self.stdio("reverse2")
        got = {}
        ts = [threading.Thread(target=lambda k=k: got.__setitem__(k, self.call(s, k, 5.0)[0])) for k in ("first", "second")]
        ts[0].start()
        time.sleep(0.2)
        ts[1].start()
        [t.join() for t in ts]
        self.assertEqual(got, {"first": ("result", "first"), "second": ("result", "second")})

    def test_a_response_for_another_id_is_ignored(self):
        s, _ = self.stdio("foreign")
        self.assertEqual(self.call(s, "mine")[0], ("result", "mine"))

    def test_ids_of_the_wrong_type_are_ignored_and_do_not_end_the_reader(self):
        for mode in ("listid", "dictid"):                 # an unhashable id used to kill the reader thread: every later call timed out
            s, _ = self.stdio(mode, timeout=1.5)
            self.assertEqual(self.call(s, "first", 1.5)[0], ("result", "first"), mode)
            self.assertEqual(self.call(s, "second", 1.5)[0], ("result", "second"), mode)

    def test_a_null_id_and_a_string_id_do_not_answer_a_request(self):
        for mode in ("nullid", "stringid"):
            s, _ = self.stdio(mode, timeout=0.6)
            self.assertEqual(self.call(s, "x", 0.6)[0][0], "McpTimeout", mode)
            self.assertEqual(len(s.transport.pending), 0)

    def test_a_boolean_id_is_not_request_1(self):
        s, _ = stdio_server("boolinit", start=1.0)
        self.addCleanup(s.close)
        self.assertFalse(s.start())                       # initialize is request 1; `true == 1` in Python
        self.assertIn("timed out", s.error)

    def test_a_float_id_that_equals_the_request_id_is_accepted_POLICY(self):
        s, _ = self.stdio("floatid")                      # JSON 3.0 for request 3: accepted by Python equality
        self.assertEqual(self.call(s)[0], ("result", "FLOAT-ID-ANSWER"))

    def test_a_second_answer_for_the_same_id_is_dropped(self):
        s, _ = self.stdio("dup")
        self.assertEqual(self.call(s, "one")[0], ("result", "one"))
        self.assertEqual(self.call(s, "two")[0], ("result", "two"))

    def test_an_answer_after_the_timeout_does_not_reach_the_next_call(self):
        s, _ = self.stdio("late", timeout=0.5)
        self.assertEqual(self.call(s, "a", 0.4, seconds=1.0)[0][0], "McpTimeout")
        time.sleep(1.3)                                   # the late answer for the first call arrives now
        self.assertEqual(self.call(s, "b", 2.0, seconds=0.1)[0], ("result", "LATE:b"))

    def test_garbage_notifications_server_requests_and_batches_around_the_answer(self):
        for mode in ("garbage", "notif", "batch"):
            s, _ = self.stdio(mode)
            self.assertEqual(self.call(s, "ok")[0], ("result", "ok"), mode)

    def test_an_answer_written_byte_by_byte(self):
        s, _ = self.stdio("slowdrip", timeout=5.0)
        self.assertEqual(self.call(s, "dripped", 5.0)[0], ("result", "dripped"))


class MalformedAnswers(StdioCase):
    def test_a_result_that_is_not_an_object_is_an_error_the_model_can_read(self):
        for mode in ("strresult", "listresult", "noresult"):
            s, _ = self.stdio(mode)
            out, _ = self.call(s)
            self.assertEqual(out[0], "McpError", mode)                 # was an AttributeError, or an empty success for noresult
            self.assertTrue(out[1], mode)
            r = mcp.McpHub.call(self.hub_of(s), "s__t", {"text": "x"})
            self.assertFalse(r["ok"], mode)
            self.assertTrue(r["text"].startswith("error:"), mode)

    def hub_of(self, s):
        hub = mcp.McpHub({}, SET)
        hub.servers = {"s": s}
        return hub

    def test_tools_that_is_not_a_list_fails_the_start(self):
        s, _ = stdio_server("badtools")
        self.addCleanup(s.close)
        box = []
        th = threading.Thread(target=lambda: box.append(s.start()), daemon=True)
        th.start()
        th.join(8)
        self.assertEqual(box, [False])
        self.assertEqual(s.status, "failed")


class Ending(StdioCase):
    def test_process_exit_ends_every_waiting_call_at_once(self):
        s, _ = self.stdio("exitnow", timeout=5.0)
        res = {}
        ts = [threading.Thread(target=lambda k=k: res.__setitem__(k, self.call(s, k, 5.0))) for k in "ab"]
        [t.start() for t in ts]
        [t.join() for t in ts]
        for out, dt in res.values():
            self.assertEqual(out[0], "McpError")
            self.assertIn("stopped", out[1])
            self.assertLess(dt, 2.5)
        self.assertEqual(len(s.transport.pending), 0)

    def test_stdout_closed_by_a_server_that_stays_alive_ends_calls_and_restarts_it(self):
        s, _ = self.stdio("eofonce", timeout=8.0)
        self.assertEqual(self.call(s, "one", 8.0)[0], ("result", "one"))
        first, dt = self.call(s, "two", 8.0)                   # races the reader's EOF: an error or, after the restart, the answer - never a timeout
        self.assertNotEqual(first[0], "McpTimeout")
        self.assertEqual(self.call(s, "three", 8.0)[0], ("result", "three"))      # a new process answers

    def test_close_while_a_call_waits_ends_the_waiter(self):
        s, _ = self.stdio("late", timeout=10.0)
        box = {}
        th = threading.Thread(target=lambda: box.update(r=self.call(s, "x", 10.0, seconds=5)), daemon=True)
        th.start()
        time.sleep(0.4)
        t0 = time.monotonic()
        s.close()
        self.assertLess(time.monotonic() - t0, 3.5)
        th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(box["r"][0][0], "McpError")


class Stopping(StdioCase):
    def test_a_stop_pressed_before_the_call_sends_nothing(self):
        s, log = self.stdio("ok")
        ev = threading.Event()
        ev.set()
        out, dt = self.call(s, "x", 5.0, ev)
        time.sleep(0.2)
        self.assertEqual(out[0], "McpCancelled")
        self.assertEqual([c for c in calls_in(log) if "call" in c], [])

    def test_the_transports_themselves_refuse_a_call_that_was_stopped_before(self):
        s, log = self.stdio("ok")
        ev = threading.Event()
        ev.set()
        with self.assertRaises(mcp.McpCancelled):
            s.transport.request("tools/call", {"name": "t", "arguments": {"text": "x"}}, 5.0, ev)
        time.sleep(0.2)
        self.assertEqual([c for c in calls_in(log) if "call" in c], [])
        self.assertEqual(len(s.transport.pending), 0)

    def test_a_stopped_call_does_not_restart_a_server_that_had_ended(self):
        s, log = self.stdio("exitnow", timeout=3.0)
        self.call(s, "x", 3.0)                                 # the server exits during this call
        old = s.transport
        ev = threading.Event()
        ev.set()
        self.assertEqual(self.call(s, "y", 3.0, ev)[0][0], "McpCancelled")
        self.assertIs(s.transport, old)                        # not started again for a call that was stopped
        self.assertEqual([c["text"] for c in calls_in(log) if "call" in c], ["x"])

    def test_a_failed_write_ends_the_call_and_leaves_nothing_pending(self):
        from unittest import mock
        s, _ = self.stdio("ok")
        with mock.patch.object(s.transport, "_send", side_effect=mcp.McpError("the server stopped")):
            out = self.call(s, "x", 3.0)
        self.assertEqual(out[0], ("McpError", "the server stopped"))
        self.assertLess(out[1], 1.0)
        self.assertEqual(len(s.transport.pending), 0)

    def test_a_call_that_returned_while_waiting_for_the_writer_is_never_sent(self):
        """a writer busy with an earlier message holds the lock; the call times out / is stopped meanwhile; when the lock is free the
        server must not receive the tool call (it would run it after the caller was told it did not)"""
        for how in ("timeout", "stop"):
            s, log = self.stdio("ok", timeout=5.0)
            ev = threading.Event()
            t = s.transport
            t.write_lock.acquire()                              # an earlier write is in progress
            box = {}
            th = threading.Thread(target=lambda: box.update(r=self.call(s, "late", 0.3 if how == "timeout" else 5.0, ev)), daemon=True)
            th.start()
            if how == "stop":
                time.sleep(0.1)
                ev.set()
            th.join(2.0)
            self.assertFalse(th.is_alive(), how)
            self.assertIn(box["r"][0][0], ("McpTimeout", "McpCancelled"), how)
            t.write_lock.release()
            time.sleep(0.5)
            self.assertEqual([c for c in calls_in(log) if "call" in c], [], how)
            self.assertEqual([c for c in calls_in(log) if "cancelled" in c], [], how)   # the server never saw the request: nothing to cancel
            self.assertEqual(self.call(s, "healthy", 3.0)[0], ("result", "healthy"), how)    # other requests are not harmed
            self.assertEqual([c["text"] for c in calls_in(log) if "call" in c], ["healthy"], how)

    def test_a_waiting_call_that_is_still_wanted_is_sent_once_the_writer_is_free(self):
        s, log = self.stdio("ok", timeout=5.0)
        t = s.transport
        t.write_lock.acquire()
        box = {}
        th = threading.Thread(target=lambda: box.update(r=self.call(s, "wanted", 5.0)), daemon=True)
        th.start()
        time.sleep(0.2)
        t.write_lock.release()
        th.join(3.0)
        self.assertEqual(box["r"][0], ("result", "wanted"))
        self.assertEqual([c["text"] for c in calls_in(log) if "call" in c], ["wanted"])

    def test_a_call_whose_write_has_begun_is_cancelled_on_the_server(self):
        s, log = self.stdio("late", timeout=5.0)
        out = self.call(s, "x", 0.4, seconds=3)
        self.assertEqual(out[0][0], "McpTimeout")
        time.sleep(0.4)
        self.assertEqual(len([c for c in calls_in(log) if "call" in c]), 1)
        self.assertEqual(len([c for c in calls_in(log) if "cancelled" in c]), 1)

    def test_a_server_that_does_not_read_stdin_cannot_hold_a_call_past_its_deadline(self):
        s, _ = self.stdio("nostdin", timeout=1.0, env={"NOSTDIN_S": "6"})
        self.addCleanup(lambda: s.transport.proc.kill())
        box = {}
        t0 = time.monotonic()

        def go():
            box["r"] = self.call(s, "x" * 3_000_000, 1.0)       # more than a pipe holds
        th = threading.Thread(target=go, daemon=True)
        th.start()
        th.join(4.0)
        self.assertFalse(th.is_alive(), "the call is still blocked in the write 4 s after a 1 s deadline")
        self.assertEqual(box["r"][0][0], "McpTimeout")
        self.assertLess(time.monotonic() - t0, 3.0)

    def test_the_stop_button_is_seen_while_the_write_is_stuck(self):
        s, _ = self.stdio("nostdin", timeout=30.0, env={"NOSTDIN_S": "6"})
        self.addCleanup(lambda: s.transport.proc.kill())
        ev = threading.Event()
        threading.Timer(0.4, ev.set).start()
        box = {}
        th = threading.Thread(target=lambda: box.update(r=self.call(s, "x" * 3_000_000, 30.0, ev)), daemon=True)
        th.start()
        th.join(4.0)
        self.assertFalse(th.is_alive())
        self.assertEqual(box["r"][0][0], "McpCancelled")

    def test_close_does_not_wait_for_a_stuck_write(self):
        s, _ = self.stdio("nostdin", timeout=1.0, env={"NOSTDIN_S": "20"})
        proc = s.transport.proc
        self.addCleanup(proc.kill)
        th = threading.Thread(target=lambda: self.call(s, "x" * 3_000_000, 1.0), daemon=True)
        th.start()
        time.sleep(0.5)
        t0 = time.monotonic()
        c = threading.Thread(target=s.close, daemon=True)
        c.start()
        c.join(8)
        self.assertFalse(c.is_alive(), "close() is still waiting for the write after 8 s")
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertIsNotNone(proc.poll())                      # the server was ended, not waited for
        th.join(3)
        self.assertFalse(th.is_alive())                        # ... which also frees the writer thread


# ------------------------------------------------------------------------------------------------ HTTP
class Fixture:
    """A loopback server; behave(fx, handler, message) answers one POST.  fx.seen: (method, lower-cased headers, message)."""

    def __init__(self, behave):
        self.behave, self.seen, self.open, self.lock, self.stop = behave, [], 0, threading.Lock(), threading.Event()
        fx = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_POST(self):
                msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with fx.lock:
                    fx.seen.append(("POST", {k.lower(): v for k, v in self.headers.items()}, msg))
                    fx.open += 1
                try:
                    fx.behave(fx, self, msg)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with fx.lock:
                        fx.open -= 1

            def do_DELETE(self):
                with fx.lock:
                    fx.seen.append(("DELETE", {k.lower(): v for k, v in self.headers.items()}, None))
                self.send_response(200)
                self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/mcp"

    def close(self):
        self.stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def reply_json(h, obj, status=200, headers=None):
    b = json.dumps(obj).encode()
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(b)))
    for k, v in (headers or {}).items():
        h.send_header(k, v)
    h.end_headers()
    h.wfile.write(b)


def reply_sse(h, chunks):
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.end_headers()
    for c in chunks:
        h.wfile.write(c.encode())
        h.wfile.flush()


def ok(rid, text="r"):
    return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}], "isError": False}}


def ev(o):
    return "event: message\ndata: " + json.dumps(o) + "\n\n"


def standard(extra):
    def behave(fx, h, msg):
        m, rid = msg.get("method"), msg.get("id")
        if m == "initialize":
            return reply_json(h, {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "fx"}}},
                              headers={"Mcp-Session-Id": "s-1"})
        if m == "tools/list":
            return reply_json(h, {"jsonrpc": "2.0", "id": rid, "result": {"tools": [{"name": "t", "inputSchema": {"type": "object"}}]}})
        if rid is None:
            h.send_response(202)
            h.end_headers()
            return
        return extra(fx, h, msg)
    return behave


class HttpCase(unittest.TestCase):
    def http(self, extra, timeout=2.0):
        fx = Fixture(standard(extra))
        s = mcp.McpServer("h", {"url": fx.url}, {**SET, "timeout_s": timeout, "start_timeout_s": 5.0})
        self.addCleanup(fx.close)
        self.addCleanup(s.close)
        self.assertTrue(s.start(), s.error)
        self.fx = fx
        return s

    def call(self, s, text="x", timeout=2.0, cancel=None):
        return timed(s.call, "t", {"text": text}, timeout, cancel)


class HttpMatching(HttpCase):
    def test_the_request_id_is_found_among_other_messages(self):
        for name, extra in (
                ("json array", lambda fx, h, m: reply_json(h, [ok(m["id"] + 50, "FOREIGN"), ok(m["id"], "right")])),
                ("sse", lambda fx, h, m: reply_sse(h, [ev({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}), ev(ok(m["id"] + 50, "FOREIGN")), ev(ok(m["id"], "right"))])),
                ("sse multi-line CRLF", lambda fx, h, m: reply_sse(h, ["event: message\r\ndata: " + json.dumps(ok(m["id"], "right"), indent=1).replace("\n", "\r\ndata: ") + "\r\n\r\n"]))):
            s = self.http(extra)
            self.assertEqual(self.call(s)[0], ("result", "right"), name)

    def test_no_matching_response_is_an_error(self):
        for name, extra in (
                ("string id", lambda fx, h, m: reply_sse(h, [ev(ok(str(m["id"]), "x"))])),
                ("stream without a response", lambda fx, h, m: reply_sse(h, [ev({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})])),
                ("unterminated last event", lambda fx, h, m: reply_sse(h, ["data: " + json.dumps(ok(m["id"], "x")) + "\n"])),
                ("a bare string", lambda fx, h, m: reply_json(h, "oops")),
                ("202 for a request", lambda fx, h, m: (h.send_response(202), h.end_headers())),
                ("not JSON", lambda fx, h, m: (h.send_response(200), h.send_header("Content-Type", "application/json"), h.end_headers(), h.wfile.write(b"<html>gateway</html>")))):
            s = self.http(extra)
            self.assertEqual(self.call(s)[0][0], "McpError", name)

    def test_a_float_id_that_equals_the_request_id_is_accepted_POLICY(self):
        s = self.http(lambda fx, h, m: reply_json(h, ok(float(m["id"]), "FLOAT")))
        self.assertEqual(self.call(s)[0], ("result", "FLOAT"))

    def test_a_result_that_is_not_an_object_is_an_error(self):
        for name, body in (("string", lambda m: {"jsonrpc": "2.0", "id": m["id"], "result": "str"}), ("neither", lambda m: {"jsonrpc": "2.0", "id": m["id"]})):
            s = self.http(lambda fx, h, m, body=body: reply_json(h, body(m)))
            self.assertEqual(self.call(s)[0][0], "McpError", name)

    def test_a_boolean_id_does_not_answer_request_1(self):
        def init_true(as_sse):
            normal = standard(lambda fx, h, m: reply_json(h, ok(m["id"])))

            def behave(fx, h, msg):                            # everything else as usual, so only the id of initialize can fail the start
                if msg.get("method") != "initialize":
                    return normal(fx, h, msg)
                body = {"jsonrpc": "2.0", "id": True, "result": {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "x"}}}
                return reply_sse(h, [ev(body)]) if as_sse else reply_json(h, body)
            return behave
        for as_sse in (False, True):
            fx = Fixture(init_true(as_sse))
            self.addCleanup(fx.close)
            s = mcp.McpServer("h", {"url": fx.url}, {**SET, "start_timeout_s": 3.0})
            self.addCleanup(s.close)
            self.assertFalse(s.start(), as_sse)                # initialize is request 1; `true == 1` in Python
            self.assertIn("did not contain a response", s.error)

    def test_the_transport_refuses_a_call_that_was_stopped_before(self):
        s = self.http(lambda fx, h, m: reply_json(h, ok(m["id"], "ran")))
        ev_ = threading.Event()
        ev_.set()
        with self.assertRaises(mcp.McpCancelled):
            s.transport.request("tools/call", {"name": "t", "arguments": {}}, 5.0, ev_)
        time.sleep(0.2)
        self.assertEqual([x for x in self.fx.seen if x[2] and x[2].get("method") == "tools/call"], [])

    def test_error_bodies(self):
        s = self.http(lambda fx, h, m: reply_json(h, {"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32000, "message": "boom"}}))
        out = self.call(s)[0]
        self.assertEqual(out[0], "McpError")
        self.assertIn("boom", out[1])
        s = self.http(lambda fx, h, m: reply_json(h, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "bad request"}}, status=400))
        out = self.call(s)[0]
        self.assertIn("400", out[1])
        self.assertIn("bad request", out[1])

    def test_session_and_protocol_headers(self):
        s = self.http(lambda fx, h, m: reply_json(h, ok(m["id"], "r")))
        self.call(s)
        s.close()
        seen = self.fx.seen
        self.assertNotIn("mcp-session-id", seen[0][1])
        self.assertNotIn("mcp-protocol-version", seen[0][1])
        for method, headers, msg in seen[1:]:
            self.assertEqual(headers.get("mcp-session-id"), "s-1", msg)
            self.assertEqual(headers.get("mcp-protocol-version"), "2025-06-18", msg)
        self.assertEqual(seen[-1][0], "DELETE")

    def test_404_ends_the_session_and_the_next_call_starts_a_new_one(self):
        n = {"k": 0}

        def extra(fx, h, m):
            n["k"] += 1
            return reply_json(h, {"error": "gone"}, status=404) if n["k"] == 1 else reply_json(h, ok(m["id"], "after restart"))
        s = self.http(extra)
        self.assertEqual(self.call(s)[0][0], "McpError")
        self.assertFalse(s.transport.alive())
        self.assertEqual(self.call(s)[0], ("result", "after restart"))
        self.assertEqual(len([x for x in self.fx.seen if x[2] and x[2].get("method") == "initialize"]), 2)

    def test_two_concurrent_calls_the_second_answered_first(self):
        gate = {"n": 0, "ev": threading.Event()}

        def extra(fx, h, m):
            gate["n"] += 1
            first = gate["n"] == 1
            if first:
                gate["ev"].wait(3)
            r = reply_json(h, ok(m["id"], "for-" + m["params"]["arguments"]["text"]))
            if not first:
                gate["ev"].set()
            return r
        s = self.http(extra, 5.0)
        got = {}
        ts = [threading.Thread(target=lambda k=k: got.__setitem__(k, self.call(s, k, 5.0)[0])) for k in ("first", "second")]
        ts[0].start()
        time.sleep(0.3)
        ts[1].start()
        [t.join() for t in ts]
        self.assertEqual(got, {"first": ("result", "for-first"), "second": ("result", "for-second")})

    def test_an_answer_after_the_timeout_does_not_reach_the_next_call(self):
        n = {"k": 0}

        def extra(fx, h, m):
            n["k"] += 1
            if n["k"] == 1:
                time.sleep(1.2)
            return reply_json(h, ok(m["id"], "for-" + m["params"]["arguments"]["text"]))
        s = self.http(extra, 0.5)
        self.assertEqual(self.call(s, "slow", 0.5)[0][0], "McpTimeout")
        time.sleep(1.2)
        self.assertEqual(self.call(s, "next", 2.0)[0], ("result", "for-next"))

    def test_the_timeout_tells_the_server(self):
        s = self.http(lambda fx, h, m: fx.stop.wait(5), 0.5)
        self.assertEqual(self.call(s, "x", 0.5)[0][0], "McpTimeout")
        time.sleep(0.6)
        sent = [x[2] for x in self.fx.seen if x[2] and x[2].get("method") == "notifications/cancelled"]
        rid = [x[2]["id"] for x in self.fx.seen if x[2] and x[2].get("method") == "tools/call"][0]
        self.assertEqual([x["params"]["requestId"] for x in sent], [rid])


class HttpStopping(HttpCase):
    def drip(self, fx, h, m):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        t0 = time.monotonic()
        while not fx.stop.is_set() and time.monotonic() - t0 < 15:
            h.wfile.write(b": keep-alive\n\n")
            h.wfile.flush()
            time.sleep(0.1)

    def test_a_stop_pressed_before_the_call_sends_nothing(self):
        s = self.http(lambda fx, h, m: reply_json(h, ok(m["id"], "ran")))
        ev_ = threading.Event()
        ev_.set()
        self.assertEqual(self.call(s, "x", 2.0, ev_)[0][0], "McpCancelled")
        time.sleep(0.3)
        self.assertEqual([x for x in self.fx.seen if x[2] and x[2].get("method") == "tools/call"], [])

    def test_an_event_stream_that_keeps_sending_is_dropped_after_the_timeout(self):
        s = self.http(self.drip, 0.6)
        self.assertEqual(self.call(s, "x", 0.6)[0][0], "McpTimeout")
        time.sleep(1.0)
        self.assertEqual(self.fx.open, 0, "the server still has the connection open: the POST thread went on reading")

    def test_an_event_stream_that_keeps_sending_is_dropped_after_the_stop_button(self):
        s = self.http(self.drip, 30.0)
        ev_ = threading.Event()
        threading.Timer(0.3, ev_.set).start()
        out, dt = self.call(s, "x", 30.0, ev_)
        self.assertEqual(out[0], "McpCancelled")
        self.assertLess(dt, 1.0)
        time.sleep(1.0)
        self.assertEqual(self.fx.open, 0)


# ------------------------------------------------------------------------------------------------ through run_with_mcp
TOOLS = [{"name": "s__t", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}]


def model_events(texts):
    text = "".join(f"<tool_call><function=s__t><parameter=text>{t}</parameter></function></tool_call>" for t in texts)
    p = frontend.OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
    out = []
    for i in range(0, len(text), 7):
        out += p.feed(text[i:i + 7])
    return out + p.finish("stop")


class Svc:
    def __init__(self, texts):
        self.texts, self.rounds, self.prepared = texts, 0, []

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None):
        self.rounds += 1
        if self.rounds == 1:
            for e in model_events(self.texts):
                yield "event", e
        else:
            yield "event", frontend.Event("content", "final answer")
        yield "done", {"finish": "stop", "completion_tokens": 1, "prompt_tokens": 1}

    def prepare(self, messages, tools, kw, max_req, cancel=None):
        self.prepared.append(copy.deepcopy(messages))
        return [1], False, 10


class ThroughRunWithMcp(unittest.TestCase):
    def hub(self, mode, timeout=2.0):
        log = tempfile.mktemp(suffix=".jsonl")
        hub = mcp.McpHub({"s": {"command": sys.executable, "args": ["-B", SCRIPTED, mode, log]}}, {"timeout_s": timeout, "start_timeout_s": 10, "max_rounds": 3})
        hub.start(wait=True)
        self.addCleanup(hub.close)
        self.assertEqual(hub.servers["s"].status, "ready", hub.servers["s"].error)
        return hub, log

    def drive(self, hub, texts, cancel=None, wrap=None):
        svc, cancel = Svc(texts), cancel or threading.Event()
        if wrap:
            wrap(hub, cancel)
        out = []
        with contextlib.redirect_stdout(io.StringIO()):
            for kind, x in server.run_with_mcp(svc, hub, [], TOOLS, {}, [1], False, 10, 10, {}, cancel, {"s__t"}):
                if kind in ("mcp", "done"):
                    out.append((kind, x))
        return svc, out

    @staticmethod
    def events(out):
        return [(x.get("event"), x.get("ok"), (x.get("text") or "")[:40]) for k, x in out if k == "mcp"]

    @staticmethod
    def finish(out):
        return [x["finish"] for k, x in out if k == "done"]

    def test_normal_call(self):
        hub, log = self.hub("ok")
        svc, out = self.drive(hub, ["hello"])
        self.assertEqual([e[0] for e in self.events(out)], ["start", "call", "result"])
        self.assertEqual([m["content"] for m in svc.prepared[0] if m["role"] == "tool"], ["hello"])
        self.assertEqual(len([c for c in calls_in(log) if "call" in c]), 1)
        self.assertEqual(self.finish(out), ["stop"])

    def test_a_malformed_result_is_an_error_result_and_the_chat_goes_on(self):
        for mode in ("strresult", "listresult", "noresult"):
            hub, log = self.hub(mode)
            svc, out = self.drive(hub, ["x"])
            results = [e for e in self.events(out) if e[0] == "result"]
            self.assertEqual(len(results), 1, mode)           # the announced call is closed in the client's view
            self.assertFalse(results[0][1], mode)
            self.assertEqual(svc.rounds, 2, mode)             # the model read the error and answered
            self.assertEqual(self.finish(out), ["stop"], mode)

    def test_timeout_is_an_error_result_and_the_tool_runs_once(self):
        hub, log = self.hub("late", 0.5)
        svc, out = self.drive(hub, ["x"])
        results = [e for e in self.events(out) if e[0] == "result"]
        self.assertFalse(results[0][1])
        self.assertIn("timed out", results[0][2])
        self.assertEqual(svc.rounds, 2)
        time.sleep(0.3)
        self.assertEqual(len([c for c in calls_in(log) if "call" in c]), 1)
        self.assertEqual(len([c for c in calls_in(log) if "cancelled" in c]), 1)

    def test_server_exit_mid_call_then_restart_on_the_next_turn(self):
        hub, log = self.hub("exitnow", 3.0)
        svc, out = self.drive(hub, ["x"])
        r = [e for e in self.events(out) if e[0] == "result"]
        self.assertFalse(r[0][1])
        self.assertIn("stopped", r[0][2])
        self.assertEqual(hub.servers["s"].status, "stopped")
        self.drive(hub, ["y"])
        self.assertEqual([c["text"] for c in calls_in(log) if "call" in c], ["x", "y"])

    def test_stop_during_a_tool_ends_the_turn_and_tells_the_server(self):
        hub, log = self.hub("late", 5.0)
        cancel = threading.Event()
        threading.Timer(0.4, cancel.set).start()
        t0 = time.monotonic()
        svc, out = self.drive(hub, ["x"], cancel)
        time.sleep(0.3)
        self.assertEqual(self.finish(out), ["cancel"])
        self.assertEqual(svc.rounds, 1)
        self.assertLess(time.monotonic() - t0, 2.5)
        self.assertEqual(len([c for c in calls_in(log) if "cancelled" in c]), 1)

    def test_a_stop_between_two_calls_of_one_answer_does_not_run_the_second_tool(self):
        hub, log = self.hub("ok")

        def wrap(h, cancel):
            real, n = h.call, {"k": 0}

            def call(name, arguments, c=None):
                r = real(name, arguments, c)
                n["k"] += 1
                if n["k"] == 1:
                    cancel.set()                              # pressed right after the first tool returned
                return r
            h.call = call
        svc, out = self.drive(hub, ["first", "second"], wrap=wrap)
        self.assertEqual([c["text"] for c in calls_in(log) if "call" in c], ["first"])
        self.assertEqual(self.finish(out), ["cancel"])


if __name__ == "__main__":
    unittest.main()
