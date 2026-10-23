"""A scripted stdio MCP server for the A10a probes.  python scripted_stdio.py MODE [LOGFILE]
It answers initialize / tools/list normally (one tool `t`), and tools/call according to MODE:
  ok         echo the call's `text` argument                       reverse2   hold two calls, answer the second first
  foreign    first a response with another id, then the real one    boolid     answer with id true (== 1 in Python)
  floatid    answer with the id as a float (3.0 for 3)  boolinit   answer initialize (id 1) with id true                          listid     first a response with id [1], then the real one
  dictid     first a response with id {"a":1}, then the real one    nullid     answer with id null (JSON-RPC parse-error shape)
  noresult   {"id":rid} without result and error                    strresult  result is a string
  listresult result is a list                                       dup        answer, then a second answer with the same id
  late       sleep ARG seconds, then answer                         garbage    a non-JSON line, then the answer
  notif      notification + server request (ping) before answer     eofafter   answer, then close stdout but stay alive    eofonce  the same, but only for the first process started
  exitnow    exit(5) on the first call                              nostdin    stop reading stdin right after tools/list (sleep NOSTDIN_S, default 8)
  stringid   answer with the id as string                           batch      answer inside a JSON array with a notification
  slowdrip   write the answer one byte at a time
The log (if given) gets one JSON line per received call."""
import json, os, sys, threading, time
mode = sys.argv[1]; logp = sys.argv[2] if len(sys.argv) > 2 else None
out_lock = threading.Lock()
def log(ev):
    if logp:
        with open(logp, "a") as f: f.write(json.dumps(ev) + "\n")
def send(o, raw=False):
    with out_lock:
        sys.stdout.write(o if raw else json.dumps(o)); sys.stdout.write("\n"); sys.stdout.flush()
def res(rid, text): return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}], "isError": False}}
held = []
for line in sys.stdin:
    if not line.strip(): continue
    m = json.loads(line); rid = m.get("id"); meth = m.get("method")
    if meth == "initialize" and mode == "dieslow":
        flag = (logp or "x") + ".started"
        if os.path.exists(flag): time.sleep(float(os.environ.get("SLOWINIT_S", "4")))
        open(flag, "w").close()
    if meth == "initialize" and mode == "boolinit" and rid == 1: send({"jsonrpc": "2.0", "id": True, "result": {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "boolid"}}}); continue
    if meth == "initialize": send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "scripted"}}}); continue
    if meth == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": 5 if mode == "badtools" else [{"name": "t", "description": "", "inputSchema": {"type": "object", "properties": {}}}]}})
        if mode == "nostdin": time.sleep(float(os.environ.get("NOSTDIN_S", "8"))); break      # stops reading stdin before any call
        continue
    if meth == "notifications/cancelled": log({"cancelled": m["params"].get("requestId")}); continue
    if meth != "tools/call": continue
    args = m["params"].get("arguments") or {}; text = str(args.get("text", "")); log({"call": rid, "text": text})
    if mode == "ok": send(res(rid, text))
    elif mode == "reverse2":
        held.append((rid, text))
        if len(held) == 2:
            for r, t in reversed(held): send(res(r, t))
            held.clear()
    elif mode == "foreign": send(res(rid + 100, "FOREIGN")); send(res(rid, text))
    elif mode == "boolid": send(res(True, "BOOL-ID-ANSWER"))
    elif mode == "floatid": send(res(float(rid), "FLOAT-ID-ANSWER"))
    elif mode == "listid": send(res([rid], "LIST-ID")); send(res(rid, text))
    elif mode == "dictid": send(res({"a": 1}, "DICT-ID")); send(res(rid, text))
    elif mode == "nullid": send(res(None, "NULL-ID"))
    elif mode == "noresult": send({"jsonrpc": "2.0", "id": rid})
    elif mode == "strresult": send({"jsonrpc": "2.0", "id": rid, "result": "just a string"})
    elif mode == "listresult": send({"jsonrpc": "2.0", "id": rid, "result": [1, 2]})
    elif mode == "dup": send(res(rid, text)); send(res(rid, "SECOND"))
    elif mode == "late":
        secs = float(args.get("seconds", 1)); threading.Timer(secs, lambda r=rid, t=text: send(res(r, "LATE:" + t))).start()
    elif mode == "garbage": send("this is not json", raw=True); send(res(rid, text))
    elif mode == "notif":
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "x"}})
        send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"}); send(res(rid, text))
    elif mode in ("eofafter", "eofonce"):
        first = not os.path.exists((logp or "x") + ".eof"); open((logp or "x") + ".eof", "w").close()
        send(res(rid, text))
        if mode == "eofafter" or first: sys.stdout.flush(); os.close(1); time.sleep(30); break      # eofonce: only the first process loses its stdout
    elif mode in ("exitnow", "dieslow"): os._exit(5)
    elif mode == "pingflood":
        for k in range(int(os.environ.get("PINGS", "4000"))): send({"jsonrpc": "2.0", "id": f"p{k}", "method": "ping"})
        send(res(rid, text))
    elif mode == "nostdin": time.sleep(float(args.get("seconds", 20))); break
    elif mode == "stringid": send(res(str(rid), text))
    elif mode == "batch": send([{"jsonrpc": "2.0", "method": "notifications/message", "params": {}}, res(rid, text)])
    elif mode == "slowdrip":
        s = json.dumps(res(rid, text)) + "\n"
        for ch in s: sys.stdout.write(ch); sys.stdout.flush(); time.sleep(0.01)
