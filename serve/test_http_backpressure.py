"""serve/test_http_backpressure.py - the HTTP layer against real loopback sockets (A09a): hang-ups, a client that stops reading,
unsent bodies.  A counting fake engine (no GPU, no pack) shows what the handler took from it and when its call ended.

Tests named ..._POLICY pin behaviour that is an open question (no write/read timeout, half-close read as a hang-up); they exist so
a change is noticed, not because it is endorsed."""
import contextlib
import io
import json
import socket
import threading
import time
import unittest
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

CYC = "abcdefghij"            # cycling bytes: the repeat-stop rule (256 equal tokens) never fires


class CountEngine(MockEngine):
    def __init__(self, tok, n_tokens=1000, delay_s=0.0):
        super().__init__(tok, "x", max_context=1 << 22, delay_s=delay_s)
        self.n, self.calls, self.lock = n_tokens, [], threading.Lock()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        rec = {"handed": 0, "end": None, "cancel_seen": False, "closed_early": False}
        with self.lock:
            self.calls.append(rec)
        try:
            for i in range(min(self.n, max_new if max_new > 0 else self.n)):
                if cancel.is_set():
                    rec["cancel_seen"] = True
                    return
                if self.delay:
                    time.sleep(self.delay)
                rec["handed"] += 1
                yield ord(CYC[i % len(CYC)])
        except GeneratorExit:
            rec["closed_early"] = True
            raise
        finally:
            rec["end"] = time.monotonic()


ROUTES = {
    "openai": ("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10 ** 6}),
    "anthropic": ("/v1/messages", {"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10 ** 6}),
    "responses": ("/v1/responses", {"model": "x", "input": "hi", "max_output_tokens": 10 ** 6}),
}


def post_bytes(route, stream=True, content_length=None, raw_body=None):
    path, body = ROUTES[route]
    b = raw_body if raw_body is not None else json.dumps({**body, "stream": stream}).encode()
    head = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(b) if content_length is None else content_length}\r\n\r\n"
    return head.encode() + b


def wait_for(cond, timeout, step=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return round(time.monotonic() - t0, 2)
        time.sleep(step)
    return None


def read_until(s, marker, timeout=8.0, limit=1 << 30):
    s.settimeout(timeout)
    buf = b""
    try:
        while len(buf) < limit and marker not in buf:
            d = s.recv(65536)
            if not d:
                break
            buf += d
    except (socket.timeout, ConnectionError):
        pass
    return buf


class HttpCase(unittest.TestCase):
    def start(self, n_tokens=1000, delay_s=0.0, monitor=False):
        tok = ByteTokenizer()
        Service.start_telemetry = lambda self: None
        self.eng = CountEngine(tok, n_tokens, delay_s)
        svc = self.svc = Service(self.eng, tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        svc.api_monitor = monitor
        self.httpd = serve(svc, port=0)
        self.port = self.httpd.server_address[1]
        self.socks = []
        self.addCleanup(self.stop)

    def stop(self):
        for s in self.socks:
            try:
                s.close()
            except OSError:
                pass
        self.httpd.shutdown()
        self.httpd.server_close()

    def connect(self, rcvbuf=None, timeout=8.0):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if rcvbuf:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        s.settimeout(timeout)
        s.connect(("127.0.0.1", self.port))
        self.socks.append(s)
        return s

    def ended(self, i=0, within=3.0):
        return wait_for(lambda: len(self.eng.calls) > i and self.eng.calls[i]["end"] is not None, within)


class Hangups(HttpCase):
    def test_a_hangup_in_a_stream_ends_the_engine_call_on_every_route(self):
        for route in ROUTES:
            self.start(n_tokens=100000, delay_s=0.002)
            c = self.connect()
            c.sendall(post_bytes(route))
            read_until(c, b"data:", 5)
            time.sleep(0.3)
            c.close()
            t = self.ended(within=3)
            self.assertIsNotNone(t, route)
            self.assertLess(t, 1.5, route)
            self.assertLess(self.eng.calls[0]["handed"], 5000, route)
            self.stop()

    def test_a_hangup_of_a_nonstreamed_request_ends_the_engine_call_on_every_route(self):
        for route in ROUTES:
            self.start(n_tokens=100000, delay_s=0.002)
            c = self.connect()
            c.sendall(post_bytes(route, stream=False))
            time.sleep(0.4)
            c.close()
            t = self.ended(within=3)
            self.assertIsNotNone(t, route)
            self.assertLess(self.eng.calls[0]["handed"], 5000, route)
            self.stop()

    def test_a_responses_request_whose_client_left_is_not_a_logged_400(self):
        """was: responses.collect raised KeyError('response') and the handler logged '400 malformed request' with a traceback"""
        self.start(n_tokens=100000, delay_s=0.002)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            c = self.connect()
            c.sendall(post_bytes("responses", stream=False))
            time.sleep(0.4)
            c.close()
            self.assertIsNotNone(self.ended(within=3))
            time.sleep(0.4)
        self.assertNotIn("malformed request", out.getvalue())
        self.assertNotIn("KeyError", out.getvalue())

    def test_the_monitor_shows_a_hangup_of_a_nonstreamed_responses_request_as_such(self):
        """with the monitor on, the record of a request whose client left says 'disconnected', not an error / HTTP 400"""
        for route in ROUTES:
            self.start(n_tokens=100000, delay_s=0.002, monitor=True)
            with contextlib.redirect_stdout(io.StringIO()):
                c = self.connect()
                c.sendall(post_bytes(route, stream=False))
                time.sleep(0.4)
                c.close()
                self.assertIsNotNone(self.ended(within=3))
                self.assertIsNotNone(wait_for(lambda: self.svc.request_records(None) and self.svc.request_records(None)[0].get("finished_at"), 3))
            rec = self.svc.request_records(None)[0]
            self.assertEqual(rec.get("state"), "disconnected", (route, rec.get("state"), rec.get("http_status"), rec.get("error")))
            self.assertNotEqual(rec.get("http_status"), 400, route)
            self.stop()

    def test_a_queued_request_whose_client_left_never_reaches_the_engine(self):
        """the single turn is held by A; B waits and hangs up; when the turn comes free B must not take it only to be cancelled
        (a native engine would be sent B's whole prompt first)"""
        self.start(n_tokens=400, delay_s=0.005)                  # A runs about 2 s
        a = self.connect()
        a.sendall(post_bytes("openai"))
        read_until(a, b"data:", 5)
        b = self.connect()
        b.sendall(post_bytes("openai", stream=False))
        time.sleep(0.3)
        b.close()
        read_until(a, b"[DONE]", 10)
        time.sleep(0.5)
        self.assertEqual(len(self.eng.calls), 1, self.eng.calls)
        c = self.connect()                                       # and the turn is not lost: the next request is served
        c.sendall(post_bytes("openai", stream=False))
        self.assertTrue(read_until(c, b"\r\n\r\n", 5).startswith(b"HTTP/1.0 200"))

    def test_a_queued_request_that_stays_is_served_after_the_first(self):
        self.start(n_tokens=200, delay_s=0.005)
        a = self.connect()
        a.sendall(post_bytes("openai"))
        read_until(a, b"data:", 5)
        b = self.connect()
        b.sendall(post_bytes("openai", stream=False))
        read_until(a, b"[DONE]", 10)
        raw = read_until(b, b"}\n", 10)
        self.assertTrue(raw.startswith(b"HTTP/1.0 200"))
        self.assertEqual(len(self.eng.calls), 2)
        self.assertEqual(self.eng.calls[1]["handed"], 200)

    def test_a_request_sent_after_a_hangup_is_served(self):
        self.start(n_tokens=100000, delay_s=0.002)
        c = self.connect()
        c.sendall(post_bytes("openai"))
        read_until(c, b"data:", 5)
        c.close()
        self.assertIsNotNone(self.ended())
        self.eng.n = 20
        d = self.connect()
        d.sendall(post_bytes("openai", stream=False))
        self.assertTrue(read_until(d, b"}\n", 5).startswith(b"HTTP/1.0 200"))


class PolicyPins(HttpCase):
    def test_a_client_that_stops_reading_holds_the_single_turn_POLICY(self):
        """no write timeout: the handler blocks in the socket write, takes no more tokens from the engine and keeps the turn;
        the next request waits until that client closes (then the call ends within a moment)"""
        self.start(n_tokens=400000)
        a = self.connect(rcvbuf=4096)
        a.sendall(post_bytes("openai"))
        read_until(a, b"data:", 5, limit=2000)
        self.assertIsNotNone(wait_for(lambda: self.eng.calls, 3))
        prev, plateau = -1, False
        for _ in range(25):                                       # taking tokens stops once the socket buffers are full
            time.sleep(0.4)
            now = self.eng.calls[0]["handed"]
            if now == prev:
                plateau = True
                break
            prev = now
        self.assertTrue(plateau)
        time.sleep(1.0)
        h2 = self.eng.calls[0]["handed"]
        self.assertEqual(h2, prev)
        self.assertLess(h2, 100000)
        self.assertIsNone(self.eng.calls[0]["end"])
        b = self.connect(timeout=2.0)
        b.sendall(post_bytes("openai", stream=False))
        self.assertEqual(read_until(b, b"\r\n\r\n", 2.0), b"")   # not served while A is stalled
        self.eng.n = 20                                           # (B's own reply is short)
        a.close()
        self.assertIsNotNone(self.ended(within=3))
        b.settimeout(5)
        self.assertTrue(read_until(b, b"\r\n\r\n", 5).startswith(b"HTTP/1.0 200"))

    def test_a_half_closed_request_is_read_as_a_hangup_POLICY(self):
        """shutdown(SHUT_WR) after the request is EOF on the server's read side, which the watcher takes for 'the client left':
        the generation is cut and the client still reads a normal-looking end (finish_reason stop, [DONE])"""
        self.start(n_tokens=300, delay_s=0.01)                   # about 3 s of output
        c = self.connect()
        c.sendall(post_bytes("openai"))
        c.shutdown(socket.SHUT_WR)
        raw = read_until(c, b"[DONE]", 10)
        self.assertIn(b"[DONE]", raw)
        self.assertLess(self.eng.calls[0]["handed"], 300)
        self.assertIn(b'"finish_reason": "stop"', raw)

    def test_unsent_bodies_and_idle_connections_hold_a_thread_each_POLICY(self):
        """no read timeout on the request socket: within the observed window nothing is given up"""
        self.start()
        base = threading.active_count()
        idle = [self.connect() for _ in range(15)]
        for _ in range(15):
            s = self.connect()
            s.sendall(post_bytes("openai", stream=False, content_length=1000, raw_body=b'{"model"'))
        time.sleep(2.0)
        self.assertGreaterEqual(threading.active_count() - base, 28)
        c = self.connect()
        c.sendall(post_bytes("openai", stream=False))              # the server still serves others meanwhile
        self.assertTrue(read_until(c, b"\r\n\r\n", 5).startswith(b"HTTP/1.0 200"))
        for s in idle + self.socks:
            s.close()
        self.assertIsNotNone(wait_for(lambda: threading.active_count() - base <= 3, 5))


class BodyErrors(HttpCase):
    def test_chunked_bodies(self):
        self.start(n_tokens=20)
        good = json.dumps({"model": "x", "messages": [{"role": "user", "content": "yo"}], "max_tokens": 3}).encode()

        def chunked(parts, tail=b"0\r\n\r\n"):
            return b"".join(f"{len(p):x}\r\n".encode() + p + b"\r\n" for p in parts) + tail
        cases = {"valid": (chunked([good[:10], good[10:]]), "200"), "bad size": (b"zz\r\nabc\r\n0\r\n\r\n", "400"),
                 "ends early": (chunked([good[:10]], tail=b""), "400"), "chunk larger than announced": (f"{len(good) - 5:x}\r\n".encode() + good + b"\r\n0\r\n\r\n", "400"),
                 "negative size": (b"-1\r\n\r\n", "400"), "huge size": (b"ffffffffff\r\nabc", "413"), "trailer cut": (chunked([good], tail=b"0\r\nX: y\r\nno-crlf"), "400")}
        for name, (body, want) in cases.items():
            c = self.connect(timeout=5)
            head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
            c.sendall(head + body)
            c.shutdown(socket.SHUT_WR)                         # the client says it has sent everything
            status = read_until(c, b"\r\n", 5).split(b"\r\n", 1)[0].decode()
            self.assertIn(want, status, name)

    def test_a_body_shorter_than_announced_then_a_close_frees_the_thread(self):
        self.start()
        base = threading.active_count()
        c = self.connect()
        c.sendall(post_bytes("openai", stream=False, content_length=1000, raw_body=b'{"model": "x"'))
        c.close()
        self.assertIsNotNone(wait_for(lambda: threading.active_count() <= base, 3))
        d = self.connect()
        d.sendall(post_bytes("openai", stream=False))
        self.assertTrue(read_until(d, b"\r\n\r\n", 5).startswith(b"HTTP/1.0 200"))

    def test_two_requests_on_one_connection_get_one_answer(self):
        self.start(n_tokens=20)
        c = self.connect()
        c.sendall(post_bytes("openai", stream=False) * 2)
        raw = read_until(c, b"never", 3)
        self.assertEqual(raw.count(b"HTTP/1.0 200"), 1)
        self.assertIn(b"Connection: close", raw)


if __name__ == "__main__":
    unittest.main()
