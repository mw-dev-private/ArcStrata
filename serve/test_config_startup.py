"""serve/test_config_startup.py - the run config and the start parameters (A11a): what the Settings view (serve/runconfig.py, POST /config)
can store, what happens to the file when an update is refused or fails, and which source wins at start (CLI / config / process env).
Private temporary config files only; server.main() runs in-process against a RECORDING engine, so no engine, hook or encoder process is
started.  Tests named ..._POLICY pin behaviour that is an open question; they exist so a change is noticed, not because it is endorsed."""
import contextlib
import http.client
import io
import json
import math
import os
import socket
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import runconfig, server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

NAN, INF = float("nan"), float("inf")
BASE = {"exe": "x", "args": ["--pack", "p", "--vram-reserve-mib", "700"], "sampling": {"temperature": 0.7, "top_p": 0.9}, "note": "kept",
        "api_key": "TEST-KEY-NOT-A-SECRET", "idle_unload_s": 30}
HUGE = 10 ** 400


class Numeric(unittest.TestCase):
    def test_values_the_settings_view_cannot_store_are_refused_with_the_key_named(self):
        """was: OverflowError (the connection dropped), or NaN / a huge whole number stored although the next start refuses or crashes on it"""
        cases = [("sampling.top_p", NAN), ("sampling.min_p", NAN), ("sampling.top_p", INF), ("sampling.top_k", INF), ("sampling.top_k", -INF), ("sampling.top_k", NAN),
                 ("reasoning_budget_tokens", INF), ("reasoning_budget_tokens", -INF), ("reasoning_budget_tokens", NAN), ("vram_reserve_mib", INF), ("vram_reserve_mib", -INF),
                 ("sampling.temperature", HUGE), ("idle_unload_s", HUGE), ("engine_silence_s", HUGE), ("sampling.temperature", -INF), ("idle_unload_s", -INF)]
        for key, value in cases:
            with self.subTest(key=key, value=repr(value)[:20]):
                with self.assertRaises(ValueError) as cm:
                    runconfig.check(key, value, BASE)
                self.assertIn(key, str(cm.exception))

    def test_normal_values_are_stored_as_before(self):
        ok = [("sampling.temperature", 0.7, 0.7), ("sampling.temperature", 0, 0.0), ("sampling.top_p", 1, 1.0), ("sampling.min_p", 0, 0.0), ("sampling.top_k", 64, 64), ("sampling.top_k", 5.0, 5),
              ("reasoning_budget_tokens", 10 ** 30, 10 ** 30), ("reasoning_budget_tokens", 1e300, int(1e300)), ("idle_unload_s", 30, 30), ("idle_unload_s", 0.5, 0.5),
              ("engine_silence_s", 0, 0), ("vram_reserve_mib", 0, 0), ("vram_reserve_mib", 4096, 4096)]
        for key, value, want in ok:
            with self.subTest(key=key, value=value):
                got = runconfig.check(key, value, BASE)
                self.assertEqual(got, want)
                self.assertIs(type(got), type(want))
        for key, bad in [("sampling.top_k", 65), ("sampling.top_k", 5.5), ("sampling.top_p", 0), ("sampling.top_p", 1.5), ("sampling.min_p", -0.1), ("idle_unload_s", -1), ("idle_unload_s", True),
                         ("idle_unload_s", "5"), ("reasoning_budget_tokens", 2.5)]:
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                runconfig.check(key, bad, BASE)

    def test_non_finite_open_ended_numbers_are_still_accepted_POLICY(self):
        """temperature / idle_unload_s / engine_silence_s take NaN and infinity today (the next start accepts them too); the file then holds the
        non-JSON tokens NaN / Infinity and GET /config sends them to the browser.  Rejecting them is a decision for the owner (see the report)."""
        for key in ("sampling.temperature", "idle_unload_s", "engine_silence_s"):
            for value in (NAN, INF):
                with self.subTest(key=key, value=value):
                    got = runconfig.check(key, value, BASE)
                    self.assertTrue(math.isnan(got) if math.isnan(value) else got == INF)

    def test_what_is_stored_is_what_the_next_start_accepts(self):
        """every value check() lets through, written and read back, passes the startup validators for that key"""
        for key, value in [("sampling.temperature", 0.3), ("sampling.top_p", 0.5), ("sampling.top_k", 7), ("sampling.min_p", 0.1), ("idle_unload_s", 12.5), ("engine_silence_s", 60)]:
            cfg, _ = runconfig.apply(BASE, {key: value})
            back = json.loads(json.dumps(cfg))
            if key.startswith("sampling."):
                self.assertEqual(server.sampling_defaults_from_config(back)[key.split(".")[1]], value)
            elif key == "engine_silence_s":
                self.assertEqual(server.engine_silence_s(back), value)
            else:
                self.assertEqual(float(back["idle_unload_s"] or 0), value)


class Env:
    def __init__(self, case):
        self.d = Path(tempfile.mkdtemp(prefix="a11t_"))
        case.addCleanup(self.close)
        self.path = self.d / "strata-test.json"
        self.path.write_text(json.dumps(BASE, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        Service.start_telemetry = lambda self: None
        self.svc = Service(MockEngine(tok, "hi", max_context=4096), tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def post(self, raw):
        with contextlib.redirect_stdout(io.StringIO()):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            try:
                c.request("POST", "/config", body=raw.encode(), headers={"Content-Type": "application/json"})
                r = c.getresponse()
                return r.status, r.read().decode()
            except (http.client.RemoteDisconnected, ConnectionError) as e:
                return None, type(e).__name__
            finally:
                c.close()

    def get(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request("GET", "/config")
            r = c.getresponse()
            return r.status, r.read().decode()
        finally:
            c.close()

    def names(self):
        return sorted(p.name for p in self.d.iterdir())

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def strict_json(text):
    try:
        json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        return True
    except ValueError:
        return False


class SettingsFile(unittest.TestCase):
    def test_a_refused_update_is_a_400_and_leaves_the_file_and_the_folder_alone(self):
        raws = ['{"set":{"sampling.top_p":NaN}}', '{"set":{"sampling.min_p":NaN}}', '{"set":{"sampling.top_k":1e309}}', '{"set":{"reasoning_budget_tokens":1e309}}',
                '{"set":{"vram_reserve_mib":-1e309}}', '{"set":{"sampling.temperature":1' + "0" * 400 + '}}', '{"set":{"idle_unload_s":1' + "0" * 400 + '}}',
                '{"set":{"sampling.top_p":0.5,"sampling.top_k":1e309}}', '{"set":{"sampling.top_k":5,"note":"x"}}']
        for raw in raws:
            with self.subTest(raw=raw[:60]):
                e = Env(self)
                before, names = e.path.read_bytes(), e.names()
                status, body = e.post(raw)
                self.assertEqual(status, 400, body[:120])
                self.assertIn("invalid_request_error", body)
                self.assertEqual(e.path.read_bytes(), before)
                self.assertEqual(e.names(), names)                       # no .bak / .tmp
                status, body = e.get()
                self.assertEqual(status, 200)
                self.assertTrue(strict_json(body))

    def test_a_valid_multi_field_update_changes_only_the_named_keys(self):
        e = Env(self)
        status, _ = e.post('{"set":{"sampling.top_k":5,"fit_max_tokens":true}}')
        cfg = json.loads(e.path.read_text())
        self.assertEqual(status, 200)
        self.assertEqual(cfg["sampling"], {"temperature": 0.7, "top_p": 0.9, "top_k": 5})
        self.assertIs(cfg["fit_max_tokens"], True)
        for k in ("exe", "args", "note", "api_key", "idle_unload_s"):
            self.assertEqual(cfg[k], BASE[k], k)
        bak = json.loads(e.path.with_name(e.path.name + ".bak").read_text())
        self.assertEqual(bak, BASE)                                      # the .bak is the file as it was
        status, _ = e.post('{"set":{"sampling.top_k":null,"sampling.top_p":null,"sampling.temperature":null}}')
        self.assertEqual(status, 200)
        self.assertNotIn("sampling", json.loads(e.path.read_text()))

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX")
    def test_the_config_and_its_backup_keep_the_permissions_of_the_file(self):
        """the config can hold the api_key: a save must not make it (or its copy) readable by others"""
        e = Env(self)
        os.chmod(e.path, 0o600)
        old = os.umask(0o002)
        try:
            self.assertEqual(e.post('{"set":{"sampling.top_k":5}}')[0], 200)
        finally:
            os.umask(old)
        self.assertEqual(stat.S_IMODE(os.stat(e.path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(e.path.with_name(e.path.name + ".bak")).st_mode), 0o600)

    def test_a_failed_write_leaves_the_config_and_no_temporary_copy(self):
        for label, patch in (("rename", mock.patch.object(os, "replace", side_effect=OSError(13, "Permission denied"))),
                             ("write", mock.patch.object(os, "write", side_effect=OSError(28, "No space left on device")))):
            with self.subTest(label):
                e = Env(self)
                before = e.path.read_bytes()
                with patch:
                    status, body = e.post('{"set":{"sampling.top_k":5}}')
                self.assertEqual(status, 500)
                self.assertIn("cannot be written", body)
                self.assertEqual(e.path.read_bytes(), before)
                self.assertFalse([n for n in e.names() if n.endswith(".tmp")], e.names())

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX")
    def test_no_copy_is_ever_more_readable_than_the_config_not_even_while_it_is_written_or_when_a_step_fails(self):
        """r02: the first version set the permissions after the copy was written (and after the first byte of the temporary file):
        a 0600 config with an api_key existed as a 0664 copy for a moment, and stayed so when the next step failed"""
        secret = json.dumps({"api_key": "SYNTHETIC-NOT-A-SECRET", "idle_unload_s": 30})

        def attempt(fail):
            d = Path(tempfile.mkdtemp(prefix="a11sv_"))
            self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
            path = d / "config.json"
            path.write_text(secret)
            path.chmod(0o600)
            old_bak = d / "config.json.bak"
            old_bak.write_text("EARLIER-BACKUP")
            old_bak.chmod(0o600)
            seen = {"writes": [], "n": {"write": 0, "fchmod": 0, "replace": 0}}
            real = {"write": os.write, "fchmod": os.fchmod, "replace": os.replace}
            seen["creates"] = []
            real_open = os.open

            def open_hook(file, flags, mode=0o777, **kw):
                if flags & os.O_CREAT:
                    seen["creates"].append(mode)                         # the mode a new file is asked for (before the umask)
                return real_open(file, flags, mode, **kw)

            def hook(kind):
                def inner(*a, **k):
                    seen["n"][kind] += 1
                    if kind == "write":
                        seen["writes"].append(stat.S_IMODE(os.fstat(a[0]).st_mode))
                    if fail == (kind, seen["n"][kind]):
                        raise OSError(28 if kind == "write" else 13, f"synthetic {kind} failure")
                    return real[kind](*a, **k)
                return inner
            old = os.umask(0o002)
            try:
                with mock.patch.object(os, "open", open_hook), mock.patch.object(os, "write", hook("write")), mock.patch.object(os, "fchmod", hook("fchmod")), mock.patch.object(os, "replace", hook("replace")):
                    try:
                        runconfig.save(path, {"api_key": "SYNTHETIC-NOT-A-SECRET", "idle_unload_s": 60})
                        err = None
                    except OSError as e:
                        err = e
            finally:
                os.umask(old)
            return d, path, old_bak, seen, err

        for fail in (None, ("write", 1), ("write", 2), ("fchmod", 1), ("fchmod", 2), ("replace", 1), ("replace", 2)):
            with self.subTest(fail=fail):
                d, path, old_bak, seen, err = attempt(fail)
                self.assertEqual(err is None, fail is None)
                for f in d.iterdir():                                       # nothing in the folder is readable by group / others
                    self.assertEqual(stat.S_IMODE(f.stat().st_mode) & 0o077, 0, f.name)
                self.assertTrue(all(m & 0o077 == 0 for m in seen["writes"]), seen["writes"])      # ... and not while a byte is written
                self.assertTrue(seen["creates"] and all(m & 0o077 == 0 for m in seen["creates"]), seen["creates"])   # ... nor is a file created open to others
                self.assertFalse([f.name for f in d.iterdir() if f.name.endswith(".tmp")])
                if fail is None:
                    self.assertEqual(json.loads(path.read_text())["idle_unload_s"], 60)
                    self.assertEqual(old_bak.read_text(), secret)
                else:
                    self.assertEqual(path.read_text(), secret)                 # the config is as it was
                    if fail[0] == "replace" and fail[1] == 2:
                        self.assertEqual(old_bak.read_text(), secret)           # the backup is a complete copy of that config
                    else:
                        self.assertEqual(old_bak.read_text(), "EARLIER-BACKUP")  # the earlier backup survives a failed save

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX")
    def test_a_wider_config_keeps_its_mode_on_the_copies(self):
        d = Path(tempfile.mkdtemp(prefix="a11sv_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        path = d / "config.json"
        path.write_text("{}")
        path.chmod(0o664)
        old = os.umask(0o022)
        try:
            runconfig.save(path, {"a": 1})
        finally:
            os.umask(old)
        for f in (path, d / "config.json.bak"):
            self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o664, f.name)

    @unittest.skipUnless(os.name == "posix", "symlinks")
    def test_a_config_that_is_a_link_stays_a_link(self):
        e = Env(self)
        real = e.d / "real.json"
        e.path.rename(real)
        e.path.symlink_to(real)
        self.assertEqual(e.post('{"set":{"sampling.top_k":5}}')[0], 200)
        self.assertTrue(e.path.is_symlink())
        self.assertEqual(json.loads(real.read_text())["sampling"]["top_k"], 5)

    def test_a_vanished_config_is_a_json_error_and_creates_nothing(self):
        e = Env(self)
        e.path.unlink()
        status, body = e.post('{"set":{"sampling.top_k":5}}')
        self.assertIn(status, (404, 500))
        self.assertEqual(e.names(), [])

    def test_accepted_non_finite_values_make_the_file_and_GET_config_non_json_POLICY(self):
        for raw in ('{"set":{"sampling.temperature":NaN}}', '{"set":{"idle_unload_s":Infinity}}'):
            with self.subTest(raw=raw):
                e = Env(self)
                self.assertEqual(e.post(raw)[0], 200)
                self.assertFalse(strict_json(e.path.read_text()))
                self.assertFalse(strict_json(e.get()[1]))                 # a browser's JSON.parse fails on it


TOK = "/home/markus/strata/data/packs/iq3_s/tokenizer"


class RecEngine(MockEngine):
    calls = []

    def __init__(self, exe, args, cwd=None, log=None, env=None, lazy=False, cpus=None, **kw):
        super().__init__(ByteTokenizer(), "x", max_context=4096)
        self.info, self.spawn, self.proc = {}, (exe, list(args)), None
        RecEngine.calls.append({"exe": exe, "args": list(args), "cwd": cwd, "lazy": lazy,
                                "env_changed": {k: v for k, v in (env or {}).items() if os.environ.get(k) != v}})

    def close(self):
        pass

    def alive(self):
        return True

    def loaded(self):
        return True


class RecVision:
    elastic = False

    def __init__(self, cfg, log=None, env=None, lazy=False, **kw):
        pass

    def alive(self):
        return True

    def shutdown(self):
        pass


@unittest.skipUnless(Path(TOK, "vocab.json").exists(), "needs the pack tokenizer files (read-only)")
class StartParameters(unittest.TestCase):
    def run_main(self, cfg, argv=(), env=None):
        d = Path(tempfile.mkdtemp(prefix="a11s_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        cp = d / "strata-test.json"
        base = {"exe": "engine/strata", "args": ["--pack", "p"], "tokenizer": TOK, "cwd": str(d)}
        base.update(cfg)
        cp.write_text(json.dumps(base), encoding="utf-8")
        RecEngine.calls.clear()
        got, opened = {}, []
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        def fake_serve(svc, host="127.0.0.1", port=0):
            got["svc"] = svc
            return mock.Mock()

        def stop(_s):
            raise KeyboardInterrupt
        saved = dict(os.environ)
        os.environ.pop("STRATA_API_KEY", None)
        os.environ.update(env or {})
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out), mock.patch.object(server, "StrataEngine", RecEngine), \
                    mock.patch.object(server, "Vision", RecVision), mock.patch.object(server, "serve", fake_serve), mock.patch.object(server.time, "sleep", stop), \
                    mock.patch("webbrowser.open", lambda url: opened.append(url)), \
                    mock.patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(cp), "--port", str(port), *argv]):
                try:
                    rc = server.main()
                except SystemExit as e:
                    rc = f"exit:{e.code}"
        finally:
            os.environ.clear()
            os.environ.update(saved)
        return {"rc": rc, "svc": got.get("svc"), "engine": RecEngine.calls[0] if RecEngine.calls else None, "opened": opened}

    def test_cli_beats_config_beats_default_including_an_explicit_zero(self):
        self.assertEqual(self.run_main({"idle_unload_s": 300}, ["--idle-unload", "0"])["svc"].idle_unload_s, 0.0)
        self.assertEqual(self.run_main({"idle_unload_s": 300})["svc"].idle_unload_s, 300.0)
        self.assertEqual(self.run_main({"idle_unload_s": 300}, ["--idle-unload", "5"])["svc"].idle_unload_s, 5.0)
        self.assertEqual(self.run_main({})["svc"].idle_unload_s, 0.0)
        self.assertEqual(self.run_main({"min_free_vram_mib": 2048}, ["--min-free-vram-mib", "0"])["svc"].min_free_vram_mib, 0)
        self.assertEqual(self.run_main({"min_free_vram_mib": 2048})["svc"].min_free_vram_mib, 2048)
        self.assertEqual(self.run_main({"before_load": "cfg"}, ["--before-load", "cli"])["svc"].before_load, "cli")

    def test_an_empty_cli_hook_does_not_switch_a_configured_one_off_POLICY(self):
        self.assertEqual(self.run_main({"before_load": "cfg"}, ["--before-load", ""])["svc"].before_load, "cfg")

    def test_switches_that_flag_and_config_both_can_set(self):
        self.assertIs(self.run_main({"fit_max_tokens": False}, ["--fit-max-tokens"])["svc"].fit_max_tokens, True)
        self.assertIs(self.run_main({"fit_max_tokens": True})["svc"].fit_max_tokens, True)
        self.assertIs(self.run_main({"fit_max_tokens": False})["svc"].fit_max_tokens, False)
        self.assertIs(self.run_main({"api_monitor": False}, ["--api-monitor"])["svc"].api_monitor, True)
        self.assertIs(self.run_main({"lazy_load": True})["engine"]["lazy"], True)
        self.assertIs(self.run_main({"lazy_load": False}, ["--lazy"])["engine"]["lazy"], True)

    def test_open_browser_only_false_is_read_POLICY(self):
        self.assertEqual(self.run_main({"open_browser": False}, ["--open"])["opened"], [])
        self.assertEqual(len(self.run_main({}, ["--open"])["opened"]), 1)
        self.assertEqual(self.run_main({"open_browser": True})["opened"], [])      # true alone opens nothing: the flag decides

    def test_api_key_order_is_cli_then_env_then_config(self):
        self.assertEqual(self.run_main({"api_key": "k-config"})["svc"].api_key, "k-config")
        self.assertEqual(self.run_main({"api_key": "k-config"}, env={"STRATA_API_KEY": "k-env"})["svc"].api_key, "k-env")
        self.assertEqual(self.run_main({"api_key": "k-config"}, ["--api-key", "k-cli"], {"STRATA_API_KEY": "k-env"})["svc"].api_key, "k-cli")

    def test_a_config_the_settings_view_could_not_have_written_refuses_the_start(self):
        r = self.run_main({"sampling": {"top_p": NAN}})
        self.assertTrue(str(r["rc"]).startswith("exit:") and "top_p" in str(r["rc"]), r["rc"])
        r = self.run_main({"sampling": {"temperature": 0.3, "top_k": 20}})
        self.assertEqual(r["svc"].sampling_defaults, {"temperature": 0.3, "top_k": 20})

    def test_the_engine_gets_the_arguments_and_environment_the_config_says(self):
        self.assertEqual(self.run_main({"args": ["--pack", "p", "--vram-reserve-mib", "0", "--vram-reserve-mib", "900"]})["engine"]["args"],
                         ["--pack", "p", "--vram-reserve-mib", "0", "--vram-reserve-mib", "900"])          # passed as written, even twice
        self.assertEqual(self.run_main({"args": ["--pack", "p", "--ple-io mmap"]})["engine"]["args"], ["--pack", "p", "--ple-io", "mmap"])
        self.assertEqual(self.run_main({"vision": {"gpu": True}})["engine"]["args"], ["--pack", "p", "--vision", "--vram-reserve-mib", "700"])
        self.assertEqual(self.run_main({"vision": {"gpu": True}, "args": ["--pack", "p", "--vram-reserve-mib", "0"]})["engine"]["args"].count("--vram-reserve-mib"), 1)
        self.assertEqual(self.run_main({"parallel": 3})["engine"]["args"], ["--pack", "p", "--batch", "3"])
        self.assertEqual(self.run_main({"parallel": 3, "args": ["--pack", "p", "--batch", "2"]})["engine"]["args"], ["--pack", "p", "--batch", "2"])
        r = self.run_main({"args": ["--pack", "p", "--reasoning-effort", "high"]})
        self.assertEqual(r["engine"]["args"], ["--pack", "p"])
        self.assertEqual(r["svc"].config_effort, "high")
        self.assertTrue(os.path.isabs(self.run_main({})["engine"]["exe"]))
        self.assertEqual(self.run_main({}, env={"A11_P": "1"})["engine"]["env_changed"], {})

    def test_config_env_is_stringified_and_beats_the_process_and_the_gpu_key_POLICY(self):
        r = self.run_main({"env": {"A11_BOOL": False, "A11_NUM": 1.5, "A11_NONE": None, "A11_STR": "x"}})
        self.assertEqual(r["engine"]["env_changed"], {"A11_BOOL": "False", "A11_NUM": "1.5", "A11_NONE": "None", "A11_STR": "x"})      # JSON false reaches the engine as the text False
        self.assertEqual(self.run_main({"env": {"A11_X": "cfg"}}, env={"A11_X": "proc"})["engine"]["env_changed"]["A11_X"], "cfg")
        self.assertEqual(self.run_main({"gpu": "0,2"})["engine"]["env_changed"]["CUDA_VISIBLE_DEVICES"], "0,2")
        self.assertEqual(self.run_main({"gpu": 1, "env": {"CUDA_VISIBLE_DEVICES": "3"}})["engine"]["env_changed"]["CUDA_VISIBLE_DEVICES"], "3")


class ServerConstants(unittest.TestCase):
    def test_config_env_does_not_set_the_servers_own_constants_POLICY(self):
        """config.env goes to the engine child only; the server's STRATA_* reads happen at import (STRATA_HTTP_BACKLOG, STRATA_ENGINE_STALL_S ...) or
        in its own process environment (STRATA_MAX_BODY_MIB): a config cannot set them"""
        child = server.child_env({"env": {"STRATA_MAX_BODY_MIB": "1", "STRATA_HTTP_BACKLOG": "9"}})
        self.assertEqual(child["STRATA_MAX_BODY_MIB"], "1")                     # the engine sees it
        self.assertNotEqual(server.body_limit(), 1 << 20)                       # the server does not
        self.assertNotIn("STRATA_HTTP_BACKLOG", os.environ)


if __name__ == "__main__":
    unittest.main()
