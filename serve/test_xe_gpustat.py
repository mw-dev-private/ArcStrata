"""serve/test_xe_gpustat.py - sycl/tools/gpustat.py (the root sampler) -> sycl/serve/xe_telemetry.py (the reader) -> serve/telemetry.py (A12d).

The two sycl files are loaded by path (they are not a package; exec'd from their source) after GPUSTAT_PCI / GPUSTAT_VRAM_MB are set, so gpustat's
import-time card search and BAR read do not run.  Every /sys and /proc path they open, list, glob, resolve or follow is remapped to a temp tree this
test builds; lspci (subprocess.run) is a fake that raises; time is a fake clock; os.getloadavg / os.cpu_count are fakes.  No hardware, no /run/gpustat.json,
no NVML / psutil, no sudo or service.  Runs with the rest of serve/: python -m unittest serve.test_xe_gpustat.

Declared contract (the files' own words + the Monitor's): a measurement that cannot be used is *unknown* (the key is absent or None) - never a made-up
number (0, a repeated string, a negative rate), and never the end of the sampler: gpustat.main() keeps running (a healthy client / series goes on), the
reader's read() returns, the Telemetry thread lives.  A healthy file reads as before (fields, units, the 100 % aggregation, the .tmp + replace write, mode 644).

Tests named ..._POLICY pin behaviour that is deliberately left to an owner decision (PCI identity of a multi-card machine, a future ts, the power window...)."""
import contextlib
import glob as real_glob_mod
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time as real_time
import types
import unittest
from pathlib import Path
from unittest import mock

from serve import telemetry as T
from serve.test_telemetry_logging import Ps, make, run_loop

SYCL = Path(__file__).resolve().parent.parent / "sycl"
REAL = dict(open=open, listdir=os.listdir, readlink=os.readlink, exists=os.path.exists, isdir=os.path.isdir, realpath=os.path.realpath,
            glob=real_glob_mod.glob)
PREFIX = ("/sys", "/proc")


class Clock:
    def __init__(self, t=5000.0):
        self.t = t

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def __getattr__(self, n):
        return getattr(real_time, n)


class Tree:
    """a fake /sys and /proc under a temp directory, and the patches that make the code under test see it"""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name + "/root"

    def close(self):
        self.tmp.cleanup()

    def R(self, p):
        return self.root + p if isinstance(p, str) and p.startswith(PREFIX) else p

    def unR(self, p):
        return p[len(self.root):] if p.startswith(self.root) else p

    @contextlib.contextmanager
    def fs(self):
        R, unR = self.R, self.unR
        with mock.patch("builtins.open", lambda p, *a, **k: REAL["open"](R(p), *a, **k)), \
                mock.patch.object(os, "listdir", lambda p=".": REAL["listdir"](R(p))), \
                mock.patch.object(os, "readlink", lambda p, **k: REAL["readlink"](R(p))), \
                mock.patch.object(os.path, "exists", lambda p: REAL["exists"](R(p))), \
                mock.patch.object(os.path, "isdir", lambda p: REAL["isdir"](R(p))), \
                mock.patch.object(os.path, "realpath", lambda p, **k: unR(REAL["realpath"](R(p)))), \
                mock.patch.object(real_glob_mod, "glob", lambda pat, **k: [unR(x) for x in REAL["glob"](R(pat), **k)]):
            yield

    def put(self, path, text):
        f = self.root + path
        os.makedirs(os.path.dirname(f), exist_ok=True)
        data = text if isinstance(text, bytes) else text.encode()
        with REAL["open"](f, "wb") as fh:
            fh.write(data)

    def rm(self, path):
        os.remove(self.root + path)

    def link(self, path, target):
        f = self.root + path
        os.makedirs(os.path.dirname(f), exist_ok=True)
        if os.path.lexists(f):
            os.remove(f)
        os.symlink(self.root + target if target.startswith("/sys") else target, f)

    def card(self, n, pci, devid="e223"):
        base = f"/sys/devices/pci0000:00/0000:00:0{n}.0/0000:0{n}:00.0"
        top = f"/sys/devices/pci0000:00/0000:00:0{n}.0"
        for d in (top, base):
            self.put(d + "/current_link_speed", "16.0 GT/s PCIe\n")
            self.put(d + "/max_link_speed", "16.0 GT/s PCIe\n")
        self.put(top + "/current_link_width", "16\n")
        self.put(top + "/max_link_width", "16\n")
        self.put(base + "/current_link_width", "1\n")
        self.put(base + "/max_link_width", "1\n")
        self.put(base + "/vendor", "0x8086\n")
        self.put(base + "/device", f"0x{devid}\n")
        self.put(base + "/tile0/gt0/freq0/act_freq", "2000\n")
        for h in (base + "/hwmon/hwmon9", "/sys/class/hwmon/hwmon9"):
            self.put(h + "/name", "xe\n")
            self.put(h + "/temp1_label", "pkg\n")
            self.put(h + "/temp1_input", "55000\n")
            self.put(h + "/energy1_input", "1000000\n")
            self.put(h + "/energy2_input", "500000\n")
            self.put(h + "/power1_cap", "200000000\n")
        self.link(f"/sys/class/drm/card{n}/device", base)
        self.link(f"/sys/bus/pci/devices/{pci}", base)
        return base

    def client(self, pid, cid, vram_kb, cyc, tot, raw=None):
        self.link(f"/proc/{pid}/fd/3", "/dev/dri/card0")
        self.put(f"/proc/{pid}/fdinfo/3", raw if raw is not None else
                 f"pos:\t0\ndrm-client-id:\t{cid}\ndrm-resident-vram0:\t{vram_kb} KiB\ndrm-cycles-ccs:\t{cyc}\ndrm-total-cycles-ccs:\t{tot}\n")
        self.put(f"/proc/{pid}/comm", "proc\n")
        self.put(f"/proc/{pid}/cgroup", "0::/\n")


class Halt(Exception):
    pass


class Env(unittest.TestCase):
    """one fake machine per test: the tree, gpustat loaded from source, the reader module loaded from source, one fake clock"""

    def setUp(self):
        self.tree = Tree()
        self.addCleanup(self.tree.close)
        self.clock = Clock()
        t = self.tree
        self.base0 = t.card(0, "0000:03:00.0")
        env = {"GPUSTAT_PCI": "0000:03:00.0", "GPUSTAT_VRAM_MB": "32768", "GPUSTAT_POWER_WINDOW": "15"}
        self.gs = types.ModuleType("gpustat_under_test")
        src = SYCL / "tools" / "gpustat.py"
        with t.fs(), mock.patch.dict(os.environ, env), mock.patch.object(subprocess, "run", side_effect=OSError("no lspci")):
            exec(compile(src.read_text(), str(src), "exec"), self.gs.__dict__)
        self.gs.OUT = t.tmp.name + "/gpustat.json"
        self.gs.time = self.clock
        self.xe = types.ModuleType("xe_telemetry_under_test")
        src = SYCL / "serve" / "xe_telemetry.py"
        exec(compile(src.read_text(), str(src), "exec"), self.xe.__dict__)
        self.xe.time = self.clock
        self.xe._XeGpu.STAT = self.gs.OUT
        self.healthy()

    def healthy(self):
        t = self.tree
        t.client(100, "c1", 2048, 1000, 10000)
        t.client(200, "c2", 1024, 500, 10000)

    def energy(self, v1=None, v2=None):
        for h in (self.base0 + "/hwmon/hwmon9", "/sys/class/hwmon/hwmon9"):
            if v1 is not None:
                self.tree.put(h + "/energy1_input", str(v1))
            if v2 is not None:
                self.tree.put(h + "/energy2_input", str(v2))

    def set_hw(self, name, text):
        for h in (self.base0 + "/hwmon/hwmon9", "/sys/class/hwmon/hwmon9"):
            self.tree.put(f"{h}/{name}", text)

    def rounds(self, n, between=None, interval=3.0):
        """gpustat.main() for n sampling rounds on the fake clock; between(i) runs before the i-th sleep (i = 1..n).  -> the list of the JSON written
        after each round; main() ending early is returned as the exception (the test then fails on it)"""
        out, seen = [], []

        def sleep(s):
            with REAL["open"](self.gs.OUT) as f:
                seen.append(json.load(f))
            i = len(seen)
            if i >= n:
                raise Halt
            if between:
                between(i)
            self.clock.t += interval
        self.gs.time = types.SimpleNamespace(time=self.clock.time, sleep=sleep)
        with self.tree.fs(), mock.patch.object(os, "getloadavg", lambda: (0.5, 0, 0)), mock.patch.object(os, "cpu_count", lambda: 8), \
                mock.patch.object(subprocess, "run", side_effect=OSError("no lspci")):
            try:
                self.gs.main()
            except Halt:
                pass
        return seen

    def reader(self, index=0, stat=None):
        with self.tree.fs():
            g = self.xe._XeGpu(index)
        g.STAT = self.gs.OUT if stat is None else stat
        return g

    def read(self, g):
        with self.tree.fs():
            return g.read()

    def write_stat(self, obj, raw=None):
        mode = "wb" if isinstance(raw, bytes) else "w"
        with REAL["open"](self.gs.OUT, mode) as f:
            f.write(raw if raw is not None else json.dumps(obj))

    def stat(self, **kw):
        d = {"name": "Intel Arc Pro B70", "pci": "0000:03:00.0", "ts": self.clock.t, "busy_pct": 71.5, "vram_used_mb": 3072, "vram_total_mb": 32768,
             "temp_pkg": 55.0, "power_w": 120.5, "power_cap_w": 200, "temp_vram_max": 61}
        d.update(kw)
        return d


# ------------------------------------------------------------------------------------------------ the reader on a sampler file
class ReaderFile(Env):
    HEALTHY = {"util": 71.5, "mem_used": 3072 * 2 ** 20, "mem_total": 32768 * 2 ** 20, "temp": 55.0, "power": 120.5, "power_limit": 200,
               "vram_temp": 61, "pcie_gen": 4, "pcie_gen_max": 4, "pcie_width": 16, "pcie_rx_mb": None, "pcie_tx_mb": None}

    def unknown_sampler_part(self, r):
        for k in ("util", "mem_used", "mem_total"):
            self.assertIsNone(r.get(k), (k, r))

    def test_healthy_file_is_read_as_before(self):
        self.write_stat(self.stat())
        g = self.reader()
        r = self.read(g)
        self.assertEqual(r, self.HEALTHY)
        self.assertEqual(self.read_name(g), "Intel Arc Pro B70")

    def read_name(self, g):
        with self.tree.fs():
            return g.name()

    def test_json_that_is_not_an_object(self):
        for raw in ("null", "[]", "[1,2]", "0", "12.5", '"text"', "true", "false", '"", {}'):
            with self.subTest(raw=raw):
                self.write_stat(None, raw)
                r = self.read(self.reader())
                self.unknown_sampler_part(r)

    def test_unreadable_files(self):
        for raw in (b"", b"{", b'{"ts": 17', b"\xff\xfe\x00garbage", b"[" * 100000, b'{"a":' * 100000):
            with self.subTest(raw=raw[:12]):
                self.write_stat(None, raw)
                self.unknown_sampler_part(self.read(self.reader()))
        os.remove(self.gs.OUT)
        self.unknown_sampler_part(self.read(self.reader()))

    def test_ts_of_the_wrong_type_or_value(self):
        for ts in (None, "x", "5000", [], {}, True, False, float("nan"), float("inf"), float("-inf"), "NaN"):
            with self.subTest(ts=ts):
                self.write_stat(self.stat(ts=ts))
                self.unknown_sampler_part(self.read(self.reader()))
        d = self.stat()
        del d["ts"]
        self.write_stat(d)
        self.unknown_sampler_part(self.read(self.reader()))
        self.write_stat(self.stat(ts=self.clock.t - 100))                          # stale: as before
        self.unknown_sampler_part(self.read(self.reader()))
        self.write_stat(self.stat(ts=self.clock.t - 14))                           # inside the 15 s: as before
        self.assertEqual(self.read(self.reader())["util"], 71.5)
        self.write_stat(self.stat(ts=int(self.clock.t)))                           # an int ts is a number too
        self.assertEqual(self.read(self.reader())["util"], 71.5)

    def test_vram_used_unusable_is_unknown_never_a_made_up_number(self):
        for v in (None, "5", "", "x", True, False, [], {}, float("nan"), float("inf")):
            with self.subTest(vram_used_mb=v):
                self.write_stat(self.stat(vram_used_mb=v))
                r = self.read(self.reader())
                self.assertNotIn("mem_used", r)
                self.assertNotIn("mem_total", r)
                self.assertEqual(r["util"], 71.5)                                  # the rest of the file is still read

    def test_vram_total_unusable_leaves_used(self):
        for v in (None, "9", "", True, [], float("nan"), float("-inf")):
            with self.subTest(vram_total_mb=v):
                self.write_stat(self.stat(vram_total_mb=v))
                r = self.read(self.reader())
                self.assertEqual(r["mem_used"], 3072 * 2 ** 20)
                self.assertIsNone(r["mem_total"])
        self.write_stat(self.stat(vram_total_mb=0))
        self.assertIsNone(self.read(self.reader())["mem_total"])                   # 0 = the BAR could not be read: unknown, as before
        d = self.stat()
        del d["vram_total_mb"]
        self.write_stat(d)
        self.assertIsNone(self.read(self.reader())["mem_total"])

    def test_vram_used_zero_and_float_are_numbers(self):
        self.write_stat(self.stat(vram_used_mb=0))
        self.assertEqual(self.read(self.reader())["mem_used"], 0)
        self.write_stat(self.stat(vram_used_mb=1.5))
        self.assertEqual(self.read(self.reader())["mem_used"], 1.5 * 2 ** 20)

    def test_other_typed_fields(self):
        # an unusable sampler value is unknown there; the reader's sysfs fallback then fills temp / power_limit exactly as for a missing value
        for key, out, fallback in (("busy_pct", "util", None), ("temp_pkg", "temp", 55.0), ("power_w", "power", None), ("power_cap_w", "power_limit", 200.0)):
            for v in (None, "7", "", True, [], {}, float("nan"), float("inf")):
                with self.subTest(key=key, value=v):
                    self.write_stat(self.stat(**{key: v}))
                    r = self.read(self.reader())
                    self.assertEqual(r.get(out), fallback)
                    self.assertEqual(r["mem_used"], 3072 * 2 ** 20)
        for v in (None, "7", True, float("nan")):
            with self.subTest(temp_vram_max=v):
                self.write_stat(self.stat(temp_vram_max=v, temp_vram=58))
                self.assertEqual(self.read(self.reader())["vram_temp"], 58)        # `max or plain`: as before, now over numbers only

    def test_name_that_is_not_text(self):
        for n in (5, ["x"], {}, True, None, ""):
            with self.subTest(name=n):
                self.write_stat(self.stat(name=n))
                self.assertEqual(self.read_name(self.reader()), "Intel Arc Pro B70")      # the PCI id's name, as for a file without one

    def test_extra_unknown_keys_are_ignored(self):
        self.write_stat(self.stat(fan_rpm=1200, host_load1=1.0, pcie={"cur": None}, energy_j=3.5))
        self.assertEqual(self.read(self.reader()), self.HEALTHY)


class ReaderSysfs(Env):
    """the fallback to the card's own sysfs files (no sampler file)"""

    def setUp(self):
        super().setUp()
        self.g = self.reader()

    def read0(self):
        return self.read(self.g)

    def test_healthy_fallback(self):
        r = self.read0()
        self.assertEqual((r["temp"], r["power_limit"], r["pcie_gen"], r["pcie_width"]), (55.0, 200.0, 4, 16))
        self.assertIsNone(r.get("power"))                                          # the first reading has no energy to compare with

    def test_power_from_two_readings(self):
        self.read0()
        self.clock.t += 2
        self.energy(v1=1000000 + 10_000_000)
        self.assertEqual(self.read0()["power"], 5.0)

    def test_temperature_text_or_empty(self):
        for text in ("N/A\n", "\n", "55.5\n", "²\n", "-\n", "0x10\n", ""):
            with self.subTest(text=text):
                self.set_hw("temp1_input", text)
                r = self.read0()
                self.assertIsNone(r["temp"])
                self.assertEqual(r["power_limit"], 200.0)                            # the other series are untouched
        self.set_hw("temp1_input", "-5000\n")
        self.assertEqual(self.read0()["temp"], -5.0)                                  # a signed integer was always accepted

    def test_binary_garbage_in_a_sysfs_file(self):
        for name in ("temp1_input", "temp1_label", "energy1_input", "power1_cap"):
            with self.subTest(file=name):
                self.set_hw(name, b"\xff\xfe\xfd")
                r = self.read0()
                self.assertIn("pcie_gen", r)
                self.set_hw(name, {"temp1_input": "55000", "temp1_label": "pkg", "energy1_input": "1000000", "power1_cap": "200000000"}[name] + "\n")

    def test_energy_and_cap_that_str_isdigit_accepts_but_int_rejects(self):
        self.set_hw("energy1_input", "²\n")
        self.set_hw("power1_cap", "²\n")
        r = self.read0()
        self.assertIsNone(r.get("power"))
        self.assertIsNone(r["power_limit"])

    def test_hwmon_vanishes_between_reads(self):
        import shutil
        shutil.rmtree(self.tree.root + self.base0 + "/hwmon")
        r = self.read0()                                                           # listdir / reads on a directory that is gone
        self.assertIn("pcie_gen", r)
        self.assertIsNone(r.get("temp"))

    def test_link_width_text(self):
        for w in ("x16\n", "\n", "²\n"):
            with self.subTest(width=w):
                self.tree.put("/sys/devices/pci0000:00/0000:00:00.0/current_link_width", w)
                r = self.read0()
                self.assertEqual(r["pcie_width"], None)                                # no port wider than x1 is known

    def test_vendor_file_that_is_not_text(self):
        self.tree.put(self.base0 + "/vendor", b"\xff\xfe")
        g = self.reader()
        self.assertFalse(g.ok())                                                   # this card is skipped, the constructor does not raise


# ------------------------------------------------------------------------------------------------ gpustat.main
class Gpustat(Env):
    def test_healthy_rounds_are_unchanged(self):
        def bump(i):
            self.tree.client(100, "c1", 2048, 1000 + 2000 * i, 10000 + 10000 * i)
            self.tree.client(200, "c2", 1024, 500 + 1000 * i, 10000 + 10000 * i)
            self.energy(v1=1000000 + 180_000_000 * i, v2=500000 + 30_000_000 * i)
        j = self.rounds(3, bump)
        self.assertEqual([x["busy_pct"] for x in j], [0.0, 30.0, 30.0])
        last = j[-1]
        self.assertEqual((last["vram_used_mb"], last["vram_total_mb"], last["name"], last["pci"]), (3, 32768, "Intel(R) Arc(TM) Pro B70 Graphics", "0000:03:00.0"))
        self.assertEqual((last["power_w"], last["pkg_power_w"], last["power_cap_w"], last["temp_pkg"], last["fan_rpm"], last["freq_mhz"]),
                         (60.0, 10.0, 200, 55.0, 0, 2000))
        self.assertEqual(last["energy_j"], 361.0)
        self.assertIsNone(j[0]["power_w"])                                          # the window is under 2.5 s in the first round
        self.assertEqual(last["pcie"]["cur"], {"gen": 4, "gts": "16.0", "width": 16})
        self.assertEqual(sorted(last), sorted(["name", "pci", "ts", "vram_used_mb", "vram_total_mb", "busy_pct", "temp_pkg", "temp_vram", "power_w", "pkg_power_w",
                                               "power_cap_w", "temp_vram_max", "temp_pcie", "temp_mctrl", "pcie", "host_load1", "host_cpus", "energy_j",
                                               "fan_rpm", "freq_mhz"]))

    def test_two_clients_at_full_load_are_clamped_to_100(self):
        def full(i):
            self.tree.client(100, "c1", 2048, 10000 * (i + 1), 10000 * (i + 1))
            self.tree.client(200, "c2", 1024, 10000 * (i + 1), 10000 * (i + 1))
        self.tree.client(100, "c1", 2048, 0, 0)
        self.tree.client(200, "c2", 1024, 0, 0)
        self.assertEqual(self.rounds(2, full)[-1]["busy_pct"], 100)

    def test_atomic_write_and_mode(self):
        self.rounds(1)
        self.assertEqual(os.stat(self.gs.OUT).st_mode & 0o777, 0o644)
        self.assertFalse(os.path.exists(self.gs.OUT + ".tmp"))                      # os.replace left no temp file
        self.assertEqual(sorted(os.listdir(os.path.dirname(self.gs.OUT) or ".")).count("gpustat.json"), 1)

    def test_unwritable_output_is_still_an_error_POLICY(self):
        """POLICY: a path that cannot be written ends main() (systemd restarts it): hardening the parser did not change the write contract"""
        self.gs.OUT = self.tree.tmp.name + "/no/such/dir/gpustat.json"
        with self.assertRaises(FileNotFoundError):
            self.rounds(1)

    # -- fdinfo
    BAD_VALUES = ("", " ", "abc", "-5 KiB", "12.5", "²", "KiB", "1e3")

    def test_unusable_fdinfo_value_skips_that_client_only(self):
        for field in ("drm-resident-vram0", "drm-cycles-ccs", "drm-total-cycles-ccs"):
            for bad in self.BAD_VALUES:
                with self.subTest(field=field, value=bad):
                    self.healthy()
                    raw = (f"drm-client-id:\tc1\ndrm-resident-vram0:\t2048 KiB\ndrm-cycles-ccs:\t1000\ndrm-total-cycles-ccs:\t10000\n")
                    raw = raw.replace(next(l for l in raw.splitlines() if l.startswith(field)), f"{field}:\t{bad}")
                    self.tree.client(100, "c1", 0, 0, 0, raw=raw)

                    def bump(i):
                        self.tree.client(200, "c2", 1024, 500 + 1000 * i, 10000 + 10000 * i)
                    j = self.rounds(3, bump)
                    self.assertEqual(len(j), 3)                                       # main() went on for every round
                    self.assertEqual(j[0]["vram_used_mb"], 1)                         # only the healthy client's 1024 KiB
                    self.assertEqual(j[-1]["busy_pct"], 10.0)                         # and its own share
                    self.setUp()                                                     # a clean machine for the next value

    def test_bad_client_that_recovers_is_measured_again(self):
        raw = "drm-client-id:\tc1\ndrm-resident-vram0:\t\ndrm-cycles-ccs:\t1000\ndrm-total-cycles-ccs:\t10000\n"
        self.tree.client(100, "c1", 0, 0, 0, raw=raw)

        def step(i):
            self.tree.client(200, "c2", 1024, 500 + 1000 * i, 10000 + 10000 * i)
            if i == 1:
                self.tree.client(100, "c1", 2048, 1000, 10000)                       # readable again
            else:
                self.tree.client(100, "c1", 2048, 1000 + 4000 * (i - 1), 10000 + 10000 * (i - 1))
        j = self.rounds(4, step)
        self.assertEqual(j[0]["vram_used_mb"], 1)
        self.assertEqual(j[1]["vram_used_mb"], 3)                                      # both clients counted again
        self.assertEqual(j[-1]["busy_pct"], 50.0)                                      # 10 % (c2) + 40 % (c1) once c1 has a baseline

    def test_fdinfo_line_without_a_value_separator(self):
        self.tree.client(100, "c1", 0, 0, 0, raw="drm-client-id\ndrm-resident-vram0:\t5 KiB\n")
        self.assertEqual(len(self.rounds(2)), 2)

    # -- cycle / total counters
    def test_cycle_counter_going_backwards_is_not_a_negative_rate(self):
        self.tree.client(100, "c1", 2048, 5000, 10000)
        self.tree.client(200, "c2", 1024, 500, 10000)

        def regress(i):
            self.tree.client(100, "c1", 2048, 100, 20000)                            # cycles fell, total rose
            self.tree.client(200, "c2", 1024, 500 + 1000 * i, 10000 + 10000 * i)
        j = self.rounds(2, regress)
        self.assertEqual(j[-1]["busy_pct"], 10.0)                                      # the healthy client only; never negative

    def test_all_intervals_unusable_is_unknown_not_zero(self):
        self.tree.client(100, "c1", 2048, 5000, 10000)
        self.tree.client(200, "c2", 1024, 5000, 10000)

        def regress(i):
            self.tree.client(100, "c1", 2048, 100, 20000)
            self.tree.client(200, "c2", 1024, 100, 20000)
        j = self.rounds(2, regress)
        self.assertIsNone(j[-1]["busy_pct"])

    def test_total_counter_going_backwards(self):
        self.tree.client(100, "c1", 2048, 5000, 10000)

        def regress(i):
            self.tree.client(100, "c1", 2048, 6000, 9000)
        j = self.rounds(2, regress)
        self.assertIsNone(j[-1]["busy_pct"])

    def test_follow_up_after_a_reset_is_correct(self):
        self.tree.client(100, "c1", 2048, 5000, 10000)

        def seq(i):
            if i == 1:
                self.tree.client(100, "c1", 2048, 100, 500)                         # restarted counters
            else:
                self.tree.client(100, "c1", 2048, 100 + 250, 500 + 1000)             # +25 % of the new interval
        j = self.rounds(3, seq)
        self.assertIsNone(j[1]["busy_pct"])
        self.assertEqual(j[2]["busy_pct"], 25.0)

    def test_no_clients_and_new_clients_are_zero_as_before(self):
        import shutil
        shutil.rmtree(self.tree.root + "/proc")
        self.assertEqual(self.rounds(2)[-1]["busy_pct"], 0.0)                         # idle: a real 0, unchanged
        self.healthy()
        self.assertEqual(self.rounds(1)[0]["busy_pct"], 0.0)                          # first round has no baseline: 0, unchanged

    # -- hwmon
    def test_hwmon_text_does_not_end_main(self):
        for name in ("temp1_input", "energy1_input", "energy2_input", "power1_cap", "fan1_input"):
            for bad in ("N/A\n", "\n", "²\n") + ((b"\xff\xfe",) if name != "fan1_input" else ()):
                with self.subTest(file=name, value=bad):
                    self.setUp()
                    self.set_hw(name, bad)
                    self.tree.put("/sys/class/hwmon/hwmon9/fan1_input", bad if name == "fan1_input" else "900\n")
                    j = self.rounds(3)
                    self.assertEqual(len(j), 3)
                    self.assertIsNone({"temp1_input": j[-1]["temp_pkg"], "energy1_input": j[-1]["power_w"], "energy2_input": j[-1]["pkg_power_w"],
                                       "power1_cap": j[-1]["power_cap_w"], "fan1_input": j[-1]["fan_rpm"]}[name])
                    if name != "temp1_input":
                        self.assertEqual(j[-1]["temp_pkg"], 55.0)                    # the other series are untouched

    def test_bad_temperature_label_is_left_out_others_kept(self):
        self.set_hw("temp2_label", "vram\n")
        self.set_hw("temp2_input", "oops\n")
        self.set_hw("temp3_label", "pcie\n")
        self.set_hw("temp3_input", "40000\n")
        j = self.rounds(1)[0]
        self.assertEqual((j["temp_pkg"], j["temp_vram"], j["temp_pcie"]), (55.0, None, 40.0))

    def test_energy_unreadable_for_a_round_then_power_over_the_longer_span(self):
        def step(i):
            if i == 1:
                self.energy(v1="N/A")
            else:
                self.energy(v1=1000000 + 3 * 60_000_000 * i)
        self.energy(v1=1000000)
        j = self.rounds(4, step)
        self.assertIsNone(j[1]["power_w"])
        self.assertIsNone(j[1]["energy_j"])
        self.assertEqual(j[2]["power_w"], 60.0)                                        # 360 J over the 6 s since the last good counter
        self.assertEqual(j[3]["power_w"], 60.0)

    def test_energy_counter_reset_is_unknown_for_that_round(self):
        def step(i):
            self.energy(v1=10 if i == 1 else 10 + 60_000_000 * (i - 1) * 3)
        self.energy(v1=900_000_000)
        j = self.rounds(3, step)
        self.assertIsNone(j[1]["power_w"])

    def test_no_hwmon_power_is_unknown_not_zero_watts(self):
        """before: a card without a readable hwmon energy counter reported power_w 0.0 (the missing counter was read as 0 and 0-0 over the window)"""
        import shutil
        shutil.rmtree(self.tree.root + "/sys/class/hwmon")
        j = self.rounds(3)
        self.assertIsNone(j[-1]["power_w"])
        self.assertIsNone(j[-1]["energy_j"])

    def test_link_width_text_at_start(self):
        self.tree.put("/sys/devices/pci0000:00/0000:00:00.0/current_link_width", "x16\n")
        self.tree.put(self.base0 + "/current_link_width", "wide\n")
        self.assertEqual(len(self.rounds(2)), 2)

    def test_find_card_with_a_vendor_file_that_is_not_text(self):
        self.tree.put(self.base0 + "/vendor", b"\xff\xfe")
        with self.tree.fs():
            self.assertIsNone(self.gs.find_card())


# ------------------------------------------------------------------------------------------------ numbers outside what a float or int() can take (r02)
HUGE = 10 ** 400                                          # does not convert to a float
LONG = "9" * 5000                                         # more digits than int() parses (sys.get_int_max_str_digits() is 4300)
U64 = 2 ** 64


def strict(test, value):
    """the answer must be strict JSON (no NaN / Infinity)"""
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, OverflowError) as e:
        test.fail(f"not strict JSON: {e}: {value!r:.200}")


class NumericReaderFile(Env):
    def test_ts_that_is_not_a_float(self):
        for ts in (HUGE, -HUGE):
            with self.subTest(ts=ts):
                self.write_stat(self.stat(ts=ts))
                r = self.read(self.reader())
                self.assertIsNone(r.get("util"))

    def test_typed_fields_that_are_not_a_float(self):
        for key, out, fallback in (("busy_pct", "util", None), ("temp_pkg", "temp", 55.0), ("power_w", "power", None), ("power_cap_w", "power_limit", 200.0),
                                   ("temp_vram_max", "vram_temp", None)):
            for v in (HUGE, -HUGE):
                with self.subTest(key=key):
                    self.write_stat(self.stat(**{key: v}))
                    r = self.read(self.reader())
                    self.assertEqual(r.get(out), fallback if key != "temp_vram_max" else None)
                    strict(self, r)

    def test_vram_used_unusable_after_conversion_too(self):
        """a value that is finite but not as bytes (1e308 MiB, 10**308 MiB) is unknown, not Infinity"""
        for v in (HUGE, -HUGE, 10 ** 308, 1e308, 1.7e308):
            with self.subTest(vram_used_mb=v):
                self.write_stat(self.stat(vram_used_mb=v))
                r = self.read(self.reader())
                self.assertNotIn("mem_used", r)
                self.assertNotIn("mem_total", r)
                self.assertEqual(r["util"], 71.5)
                strict(self, r)

    def test_vram_total_unusable_after_conversion_leaves_used(self):
        for v in (HUGE, 10 ** 308, 1e308):
            with self.subTest(vram_total_mb=v):
                self.write_stat(self.stat(vram_total_mb=v))
                r = self.read(self.reader())
                self.assertEqual(r["mem_used"], 3072 * 2 ** 20)
                self.assertIsNone(r["mem_total"])
                strict(self, r)

    def test_large_valid_vram_is_kept(self):
        self.write_stat(self.stat(vram_used_mb=1e300, vram_total_mb=10 ** 20))
        r = self.read(self.reader())
        self.assertEqual(r["mem_used"], 1e300 * 2 ** 20)
        self.assertEqual(r["mem_total"], 10 ** 20 * 2 ** 20)
        strict(self, r)


class NumericSysfs(Env):
    def setUp(self):
        super().setUp()
        self.g = self.reader()

    def read0(self):
        r = self.read(self.g)
        strict(self, r)
        return r

    def test_counters_beyond_u64_or_too_long_are_unknown(self):
        for text in (LONG, "9" * 21, "9" * 25, str(U64), "0" * 30 + "1"):
            for name, key in (("energy1_input", "power"), ("power1_cap", "power_limit")):
                with self.subTest(file=name, digits=len(text)):
                    self.setUp()
                    self.set_hw(name, text + "\n")
                    self.read0()
                    self.clock.t += 2
                    r = self.read0()
                    self.assertIsNone(r.get(key))

    def test_largest_u64_counter_is_a_number(self):
        self.set_hw("energy1_input", str(U64 - 1) + "\n")
        self.set_hw("power1_cap", str(U64 - 1) + "\n")
        r = self.read0()
        self.assertEqual(r["power_limit"], (U64 - 1) / 1e6)

    def test_temperature_that_overflows_the_division(self):
        for text in (LONG, "9" * 400, "-" + "9" * 400, "-" + LONG, "9" * 21, str(U64)):
            with self.subTest(digits=len(text)):
                self.set_hw("temp1_input", text + "\n")
                r = self.read0()
                self.assertIsNone(r["temp"])
                self.assertEqual(r["power_limit"], 200.0)
        self.set_hw("temp1_input", "-" + str(U64 - 1) + "\n")
        self.assertEqual(self.read0()["temp"], -(U64 - 1) / 1000)

    def test_link_width_with_a_long_number(self):
        for text in (LONG, "9" * 25):
            with self.subTest(digits=len(text)):
                self.tree.put("/sys/devices/pci0000:00/0000:00:00.0/current_link_width", text + "\n")
                self.assertIsNone(self.read0()["pcie_width"])


class NumericGpustat(Env):
    def run3(self, between=None):
        j = self.rounds(3, between)
        self.assertEqual(len(j), 3)                                    # main() went on
        for x in j:
            strict(self, x)
        return j

    def test_hwmon_counters_with_huge_or_long_digits(self):
        for text in (LONG, "9" * 21, "9" * 400, str(U64)):
            for name, key in (("energy1_input", "power_w"), ("energy2_input", "pkg_power_w"), ("power1_cap", "power_cap_w"), ("fan1_input", "fan_rpm"),
                              ("temp1_input", "temp_pkg")):
                with self.subTest(file=name, digits=len(text)):
                    self.setUp()
                    self.set_hw(name, text + "\n")
                    j = self.run3()
                    self.assertIsNone(j[-1][key])
                    if name != "temp1_input":
                        self.assertEqual(j[-1]["temp_pkg"], 55.0)

    def test_negative_temperature_text_with_a_long_number(self):
        self.set_hw("temp2_label", "vram\n")
        self.set_hw("temp2_input", "-" + LONG + "\n")
        j = self.run3()
        self.assertIsNone(j[-1]["temp_vram"])
        self.assertEqual(j[-1]["temp_pkg"], 55.0)

    def test_freq_and_link_width_with_long_digits(self):
        self.tree.put(self.base0 + "/tile0/gt0/freq0/act_freq", LONG + "\n")
        self.tree.put(self.base0 + "/current_link_width", LONG + "\n")
        j = self.run3()
        self.assertIsNone(j[-1]["freq_mhz"])

    def test_fdinfo_numbers_beyond_u64_skip_only_that_client(self):
        for field in ("drm-resident-vram0", "drm-cycles-ccs", "drm-total-cycles-ccs"):
            for bad in (LONG, "9" * 21, str(U64), "-" + str(U64)):
                with self.subTest(field=field, digits=len(bad)):
                    self.setUp()
                    raw = "drm-client-id:\tc1\ndrm-resident-vram0:\t2048 KiB\ndrm-cycles-ccs:\t1000\ndrm-total-cycles-ccs:\t10000\n"
                    raw = raw.replace(next(l for l in raw.splitlines() if l.startswith(field)), f"{field}:\t{bad}")
                    self.tree.client(100, "c1", 0, 0, 0, raw=raw)

                    def bump(i):
                        self.tree.client(200, "c2", 1024, 500 + 1000 * i, 10000 + 10000 * i)
                    j = self.run3(bump)
                    self.assertEqual(j[0]["vram_used_mb"], 1)
                    self.assertEqual(j[-1]["busy_pct"], 10.0)

    def test_largest_valid_counters_still_work(self):
        big = U64 - 1
        self.tree.client(100, "c1", 0, 0, 0, raw=f"drm-client-id:\tc1\ndrm-resident-vram0:\t{big} KiB\ndrm-cycles-ccs:\t0\ndrm-total-cycles-ccs:\t0\n")
        j = self.run3()
        self.assertEqual(j[0]["vram_used_mb"], (big + 1024) // 1024)               # c1's u64 maximum + the healthy c2's 1024 KiB, in whole MiB
        strict(self, j[0])


# ------------------------------------------------------------------------------------------------ gpustat -> reader -> Telemetry
class Adapter:
    """the real reader, called inside the fake file tree"""

    def __init__(self, env, g):
        self.env, self.g = env, g
        self.note = None

    def ok(self):
        return self.g.ok()

    def name(self):
        with self.env.tree.fs():
            return self.g.name()

    def read(self):
        with self.env.tree.fs():
            return self.g.read()


class Together(Env):
    def sampler(self):
        g = Adapter(self, self.reader())
        t = make(g, ps=Ps())
        patch = mock.patch.object(T, "time", self.clock)
        patch.start()
        self.addCleanup(patch.stop)
        return t

    def test_healthy_output_reaches_the_snapshot_unchanged(self):
        j = self.rounds(2, lambda i: self.tree.client(100, "c1", 2048, 3000, 20000))
        self.assertEqual(j[-1]["busy_pct"], 20.0)
        t = run_loop(self.sampler(), 1)
        now = t.snapshot()["now"]
        self.assertEqual(now["gpu_util"], 20.0)
        self.assertEqual(now["gpu_mem_used"], j[-1]["vram_used_mb"] * 2 ** 20)
        self.assertEqual(now["gpu_mem_total"], 32768 * 2 ** 20)
        self.assertEqual(now["gpu_temp"], 55.0)
        self.assertEqual(t.snapshot()["static"]["gpu_name"], "Intel(R) Arc(TM) Pro B70 Graphics")
        self.assertEqual(t.snapshot()["history"]["gpu_util"], [20.0])

    def test_broken_output_is_a_missing_measurement_not_a_dead_thread(self):
        t = self.sampler()
        self.write_stat(self.stat())
        raws = [None, "null", "[]", '{"ts": null}', json.dumps(self.stat(vram_used_mb="5")), json.dumps(self.stat(vram_total_mb=None)), b"\xff", "{"]
        seq = iter(raws)

        def between():
            raw = next(seq, None)
            if raw is None:
                self.write_stat(self.stat(busy_pct=33.0))                            # the sampler is healthy again
            elif isinstance(raw, str) and raw.startswith("{") and raw.endswith("}"):
                self.write_stat(None, raw)
            else:
                self.write_stat(None, raw if raw is not None else "null")
        self.write_stat(self.stat())
        run_loop(t, len(raws) + 2, between)
        h = t.snapshot()["history"]
        self.assertEqual(len(h["cpu"]), len(raws) + 2)                              # a sample for every second: the loop never stopped
        self.assertEqual(h["gpu_util"][0], 71.5)
        self.assertEqual(h["gpu_util"][-1], 33.0)                                    # the next good file is shown at once
        self.assertEqual(t.snapshot()["now"]["gpu_util"], 33.0)

    def test_unusable_vram_never_becomes_a_number_in_the_snapshot(self):
        t = self.sampler()
        for bad in ("5", True, None, [], float("nan")):
            self.write_stat(self.stat(vram_used_mb=bad))
            now = t.sample()
            self.assertNotIn("gpu_mem_used", now)
            self.assertEqual(now["gpu_util"], 71.5)

    def test_nan_busy_is_unknown_in_the_snapshot(self):
        t = self.sampler()
        self.write_stat(self.stat(busy_pct=float("nan"), power_w=float("inf")))
        now = t.sample()
        self.assertIsNone(now["gpu_util"])
        self.assertIsNone(now["gpu_power"])
        json.dumps(now, allow_nan=False)                                             # strict JSON: no NaN / Infinity from the typed Xe fields

    def test_gpustat_to_telemetry_with_a_bad_client_and_a_negative_rate_candidate(self):
        self.tree.client(100, "c1", 2048, 5000, 10000)
        self.tree.client(200, "c2", 1024, 0, 0, raw="drm-client-id:\tc2\ndrm-resident-vram0:\t\n")

        def regress(i):
            self.tree.client(100, "c1", 2048, 100, 20000)
        j = self.rounds(2, regress)
        self.assertIsNone(j[-1]["busy_pct"])
        t = run_loop(self.sampler(), 1)
        now = t.snapshot()["now"]
        self.assertIsNone(now["gpu_util"])                                           # unknown - not 0 and not negative
        self.assertEqual(now["gpu_mem_used"], 2 * 2 ** 20)                           # the healthy client's 2048 KiB

    def test_numbers_outside_float_range_are_unknown_in_the_snapshot_and_the_thread_lives(self):
        t = self.sampler()
        files = [self.stat(ts=HUGE), self.stat(vram_used_mb=HUGE), self.stat(vram_used_mb=1e308), self.stat(vram_total_mb=1e308),
                 self.stat(busy_pct=HUGE, power_w=-HUGE), self.stat()]
        it = iter(files[1:])
        self.write_stat(files[0])
        run_loop(t, len(files), lambda: self.write_stat(next(it, files[-1])))
        h = t.snapshot()["history"]
        self.assertEqual(len(h["cpu"]), len(files))                                 # a sample for every file, no thread death
        self.assertEqual(h["gpu_util"][-1], 71.5)
        self.assertEqual(h["gpu_mem_used"], [None, None, None, 3072 * 2 ** 20, 3072 * 2 ** 20, 3072 * 2 ** 20])   # a total that cannot be bytes leaves `used`
        self.assertEqual(t.snapshot()["now"]["gpu_mem_total"], 32768 * 2 ** 20)
        strict(self, t.snapshot()["now"])

    # -- POLICY pins: decided elsewhere (proposal sheet)
    def test_second_card_shows_the_file_card_POLICY(self):
        """POLICY (X-1): gpustat covers one card (GPUSTAT_PCI / the first Intel xe card) and writes its PCI address as `pci`; the reader of card
        index 1 reads the same file and does not compare `pci` with its own device, so it shows the file card's load and VRAM."""
        self.tree.card(1, "0000:04:00.0")
        self.write_stat(self.stat())
        r0, r1 = self.read(self.reader(0)), self.read(self.reader(1))
        self.assertEqual((r0["util"], r0["mem_used"]), (r1["util"], r1["mem_used"]))

    def test_legacy_file_without_pci_is_read_POLICY(self):
        d = self.stat()
        del d["pci"]
        self.write_stat(d)
        self.assertEqual(self.read(self.reader())["util"], 71.5)

    def test_future_ts_counts_as_fresh_POLICY(self):
        """POLICY (X-5): the reader only checks `now - ts < 15`; a ts far in the future (a clock step) passes it"""
        self.write_stat(self.stat(ts=self.clock.t + 1e6))
        self.assertEqual(self.read(self.reader())["util"], 71.5)

    def test_sysfs_power_has_no_window_POLICY(self):
        """POLICY (X-4): without the sampler file the reader's power is the delta of two readings (gpustat averages over POWER_WINDOW s because
        the counters update in bursts): an unchanged counter between two readings is 0.0 W"""
        g = self.reader()
        self.read(g)
        self.clock.t += 1
        self.assertEqual(self.read(g)["power"], 0.0)


if __name__ == "__main__":
    unittest.main()
