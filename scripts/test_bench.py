import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
# bench-spectral.py imports its helpers from this module.
sys.modules["bench"] = bench
spec = importlib.util.spec_from_file_location("bench_spectral",
                                              Path(__file__).with_name("bench-spectral.py"))
spectral = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectral)


class BuildIdentity(unittest.TestCase):
    def test_custom_target_never_executes_stale_default_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "Cargo.toml").write_text('[package]\nname="probe"\nversion="0.1.0"\nedition="2024"\n')
            (root / "src/main.rs").write_text('fn main() { println!("fresh"); }\n')
            subprocess.run(["cargo", "generate-lockfile", "--offline"], cwd=root, check=True, capture_output=True)
            stale = root / "target/release/probe"
            stale.parent.mkdir(parents=True)
            stale.write_text('#!/bin/sh\necho stale\n')
            stale.chmod(0o755)
            version = subprocess.check_output(["rustc", "-vV"], text=True)
            host = next(line.removeprefix("host: ") for line in version.splitlines() if line.startswith("host: "))
            old_root = bench.ROOT
            bench.ROOT = root
            try:
                binary = bench.build_binary("probe", dict(os.environ,
                    CARGO_TARGET_DIR=str(root / "custom"), CARGO_BUILD_TARGET=host), root / "build.log")
            finally:
                bench.ROOT = old_root
            self.assertEqual(subprocess.check_output([binary], text=True).strip(), "fresh")


class Arguments(unittest.TestCase):
    def parse(self, *arguments):
        parser = bench.argument_parser()
        args = parser.parse_args(arguments)
        bench.check_bounds(parser, args)
        return args

    def rejection(self, *arguments):
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            self.parse(*arguments)
        return error.getvalue()

    def test_defaults_of_the_existing_workloads_are_unchanged(self):
        args = self.parse()
        self.assertEqual((args.samples, args.warmup, args.elements), (20, 5, 1_000_003))
        self.assertEqual(args.matrix, [129, 131, 127])

    def test_gemm_defaults_to_1024_cubed_and_takes_m_n_k(self):
        self.assertEqual(self.parse().gemm, [1024, 1024, 1024])
        self.assertEqual(self.parse("--gemm", "256", "128", "64").gemm, [256, 128, 64])
        self.assertEqual(self.parse("--gemm", "8192", "8192", "8192").gemm, [8192] * 3)

    def test_gemm_rejects_a_dimension_above_8192_and_malformed_shapes(self):
        self.assertIn("gemm dimensions must be <= 8192", self.rejection("--gemm", "1", "8193", "1"))
        for shape in (["0", "1", "1"], ["1", "1"], ["1", "1", "1", "1"]):
            self.rejection("--gemm", *shape)

    def test_gemm_repeat_defaults_to_64_and_stays_from_1_to_256(self):
        self.assertEqual(self.parse().gemm_repeat, 64)
        for repeat in (1, 256):
            self.assertEqual(self.parse("--gemm-repeat", str(repeat)).gemm_repeat, repeat)
        self.assertIn("gemm repeat must be <= 256", self.rejection("--gemm-repeat", "257"))
        self.rejection("--gemm-repeat", "0")

    def test_backends_default_to_both_and_take_a_subset_in_runtime_order(self):
        self.assertEqual(self.parse().backends, ["classic", "metal4"])
        self.assertEqual(self.parse("--backends", "classic").backends, ["classic"])
        self.assertEqual(self.parse("--backends", "metal4").backends, ["metal4"])
        self.assertEqual(self.parse("--backends", "metal4,classic").backends, ["classic", "metal4"])

    def test_backends_rejects_unknown_names_empty_lists_and_repeats(self):
        for value in ("cuda", "classic,vulkan", "", "classic,", "classic,classic"):
            self.assertIn("distinct backends", self.rejection("--backends", value))

    def test_mlx_is_opt_in(self):
        self.assertFalse(self.parse().mlx)
        self.assertTrue(self.parse("--mlx").mlx)

    def test_cases_default_to_every_workload_and_take_a_subset_in_workload_order(self):
        self.assertEqual(self.parse().cases, list(bench.WORKLOADS))
        self.assertEqual(self.parse("--cases", "sort").cases, ["sort"])
        self.assertEqual(self.parse("--cases", "sort,vec-add").cases, ["vec-add", "sort"])
        self.assertEqual(self.parse("--cases", "gemm,fft,sort").cases, ["sort", "fft", "gemm"])
        self.assertEqual(self.parse("--cases", "gemm,stencil,fft").cases,
                         ["fft", "stencil", "gemm"])

    def test_cases_rejects_unknown_names_empty_lists_and_repeats(self):
        for value in ("bitonic", "sort,bitonic", "histogram", "", "sort,", "sort,sort"):
            self.assertIn("distinct workloads", self.rejection("--cases", value))

    def test_mlx_needs_the_gemm_workload(self):
        self.assertIn("--mlx measures the gemm workload",
                      self.rejection("--cases", "sort", "--mlx"))
        self.assertEqual(self.parse("--cases", "gemm", "--mlx").cases, ["gemm"])

    def test_sort_defaults_to_2_to_the_20_keys_and_stays_from_1_to_2_to_the_24(self):
        self.assertEqual(self.parse().sort, 1 << 22)
        for keys in (1, 1 << 24):
            self.assertEqual(self.parse("--sort", str(keys)).sort, keys)
        self.assertIn("sort must be <= 16777216", self.rejection("--sort", str((1 << 24) + 1)))
        self.rejection("--sort", "0")

    def test_fft_defaults_to_8192_rows_of_4096_and_takes_n_and_rows(self):
        self.assertEqual(self.parse().fft, [4096, 8192])
        self.assertEqual(self.parse("--fft", "2", "1").fft, [2, 1])
        self.assertEqual(self.parse("--fft", str(1 << 26), "1").fft, [1 << 26, 1])
        self.assertEqual(self.parse("--fft", "1024", str(1 << 16)).fft, [1024, 1 << 16])

    def test_fft_needs_a_power_of_two_n_from_2_and_at_most_2_to_the_26_values(self):
        for shape in (["1", "1"], ["3", "1"], ["4095", "2"], ["4096", str((1 << 14) + 1)]):
            self.assertIn("fft n must be a power of two >= 2 and n * rows <= 67108864",
                          self.rejection("--fft", *shape))
        for shape in (["0", "1"], ["4096", "0"], ["4096"], ["4096", "1", "1"]):
            self.rejection("--fft", *shape)

    def test_stencil_defaults_to_2048_iterations_on_2048_squared_and_takes_a_shape(self):
        self.assertEqual(self.parse().stencil, [2048, 2048, 2048])
        self.assertEqual(self.parse("--stencil", "1", "1", "1").stencil, [1, 1, 1])
        self.assertEqual(self.parse("--stencil", "4096", "4096", "4096").stencil, [4096] * 3)
        self.assertEqual(self.parse("--stencil", str(1 << 24), "1", "8").stencil, [1 << 24, 1, 8])

    def test_stencil_needs_at_most_2_to_the_24_cells_and_4096_iterations(self):
        for shape in (["4097", "4096", "1"], ["1", str((1 << 24) + 1), "1"], ["1", "1", "4097"]):
            self.assertIn("stencil must have width * height <= 16777216 and iterations <= 4096",
                          self.rejection("--stencil", *shape))
        for shape in (["0", "1", "1"], ["1", "1", "0"], ["16", "16"], ["16", "16", "1", "1"]):
            self.rejection("--stencil", *shape)

    def test_established_bounds_keep_their_messages(self):
        self.assertIn("elements must be <= 16000000", self.rejection("--elements", "16000001"))
        self.assertIn("matrix dimensions must be <= 1024", self.rejection("--matrix", "1", "1025", "1"))


class Workloads(unittest.TestCase):
    def test_every_workload_names_its_kernel_source_and_reference(self):
        for package, (kernels, reference) in bench.WORKLOADS.items():
            for source in (kernels, reference):
                self.assertTrue((bench.ROOT / source).is_file(), f"{package}: {source}")

    def test_gemm_compares_the_gemm_package_with_its_msl_twin(self):
        self.assertEqual(bench.WORKLOADS["gemm"], ("examples/gemm/host/src/main.rs",
                                                   "benchmarks/reference-msl/gemm.metal"))


PROFILE = json.dumps({"SPDisplaysDataType": [{
    "_name": "Apple M4 Max",
    "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
    "spdisplays_ndrvs": [{"_name": "Color LCD", "spdisplays_main": "spdisplays_yes"}],
    "spdisplays_vendor": "sppci_vendor_Apple",
    "sppci_bus": "spdisplays_builtin",
    "sppci_cores": "32",
    "sppci_device_type": "spdisplays_gpu",
    "sppci_model": "Apple M4 Max"}]})
SYSCTL = {"machdep.cpu.brand_string": "Apple M4 Max", "hw.model": "Mac16,6",
          "hw.memsize": "38654705664"}


def answer(sysctl=SYSCTL, profile=PROFILE):
    def query(args):
        if args[0] == "sysctl":
            return sysctl.get(args[2])
        return profile if args[:2] == ["system_profiler", "SPDisplaysDataType"] else None
    return query


class Machine(unittest.TestCase):
    def test_gpu_cores_come_from_the_first_gpu(self):
        self.assertEqual(bench.gpu_cores(PROFILE), 32)
        two = json.dumps({"SPDisplaysDataType": [{"sppci_cores": "10"}, {"sppci_cores": "20"}]})
        self.assertEqual(bench.gpu_cores(two), 10)
        self.assertEqual(bench.gpu_cores('{"SPDisplaysDataType": [{"sppci_cores": 8}]}'), 8)

    def test_gpu_cores_are_null_without_a_usable_count(self):
        profiles = (None, "", "not json", "{}", "[]", '{"SPDisplaysDataType": []}',
                    '{"SPDisplaysDataType": [{"_name": "Intel UHD Graphics"}]}',
                    '{"SPDisplaysDataType": [{"sppci_cores": "many"}]}',
                    '{"SPDisplaysDataType": [{"sppci_cores": "0"}]}')
        for profile in profiles:
            self.assertIsNone(bench.gpu_cores(profile), profile)

    def test_identity_joins_the_sysctl_values_and_the_profile(self):
        self.assertEqual(bench.machine_identity(answer()),
                         {"chip": "Apple M4 Max", "model": "Mac16,6",
                          "memory_bytes": 38_654_705_664, "gpu_cores": 32})

    def test_a_host_without_these_tools_gets_nulls(self):
        nulls = {"chip": None, "model": None, "memory_bytes": None, "gpu_cores": None}
        self.assertEqual(bench.machine_identity(lambda args: None), nulls)

    def test_each_missing_value_is_null_on_its_own(self):
        partial = bench.machine_identity(answer(sysctl={"hw.model": "Mac16,6"}, profile="{}"))
        self.assertEqual(partial, {"chip": None, "model": "Mac16,6",
                                   "memory_bytes": None, "gpu_cores": None})
        for memory in ("lots", "0", "-1"):
            identity = bench.machine_identity(answer(sysctl=dict(SYSCTL, **{"hw.memsize": memory})))
            self.assertIsNone(identity["memory_bytes"], memory)

    def test_tool_output_is_null_for_a_missing_failing_or_silent_tool(self):
        self.assertIsNone(bench.tool_output(["metal-oxide-no-such-tool"]))
        self.assertIsNone(bench.tool_output([sys.executable, "-c", "raise SystemExit(3)"]))
        self.assertIsNone(bench.tool_output([sys.executable, "-c", "pass"]))
        self.assertEqual(bench.tool_output([sys.executable, "-c", "print(' x ')"]), "x")


class Statistics(unittest.TestCase):
    def test_summary_follows_the_runners_ranks(self):
        self.assertEqual(bench.summarize([100, 3, 1, 4, 2]),
                         {"min": 1, "p10": 1, "median": 3, "p90": 100, "max": 100, "mean": 22.0})
        self.assertEqual(bench.summarize([40, 10, 30, 20]),
                         {"min": 10, "p10": 10, "median": 25, "p90": 40, "max": 40, "mean": 25.0})
        summary = bench.summarize(range(11))
        self.assertEqual((summary["p10"], summary["median"], summary["p90"]), (1, 5, 9))
        self.assertEqual(bench.summarize([7])["p90"], 7)

    def test_per_gemm_time_rounds_half_up_like_the_runner(self):
        self.assertEqual([bench.per_gemm_ns(ns, 8) for ns in (2499, 2500, 2503, 2504)],
                         [312, 313, 313, 313])
        self.assertEqual(bench.per_gemm_ns(7, 1), 7)


def mlx_result(**changes):
    """An MLX result for 256^3, R = 8 and two samples, with `changes` to its case."""
    case = {"name": "gemm-f32", "variant": "mlx", "shape": [256, 256, 256], "repeat": 8,
            "samples": [{"host_ns": 800, "host_per_gemm_ns": 100}] * 2}
    return {"version": "0.32.4", "python": "3.12.12", "device": "Apple M4 Max",
            "cases": [dict(case, **changes)]}


class Merge(unittest.TestCase):
    def report(self):
        return {"config": {"gemm": [256, 256, 256], "gemm_repeat": 8, "samples": 2},
                "backends": [{"backend": "classic"}, {"backend": "metal4"}, {"backend": "mps"}],
                "timing": {"unit": "ns", "mps": "note"}}

    def test_mlx_joins_the_report_beside_the_runner_array(self):
        report, mlx = self.report(), mlx_result()
        merged = bench.merge_mlx(report, mlx)
        self.assertEqual(merged["mlx"], mlx)
        self.assertEqual(merged["backends"], self.report()["backends"])
        self.assertEqual(merged["timing"], dict(self.report()["timing"], mlx=bench.MLX_TIMING))
        self.assertEqual(report, self.report())

    def test_the_note_says_that_mlx_has_no_gpu_timestamps(self):
        self.assertIn("exposes no GPU timestamps", bench.MLX_TIMING)
        self.assertIn("with the other backends' host_per_gemm_ns", bench.MLX_TIMING)
        self.assertIn("not their gpu_per_gemm_ns", bench.MLX_TIMING)

    def test_the_note_says_that_mlx_matmuls_may_overlap(self):
        self.assertIn("independent", bench.MLX_TIMING)
        self.assertIn("may overlap", bench.MLX_TIMING)

    def test_a_result_for_another_shape_repeat_or_sample_count_is_refused(self):
        for changes in ({"shape": [256, 256, 128]}, {"repeat": 4},
                        {"samples": [{"host_ns": 800, "host_per_gemm_ns": 100}]}):
            with self.assertRaises(RuntimeError, msg=changes):
                bench.merge_mlx(self.report(), mlx_result(**changes))


class Timeouts(unittest.TestCase):
    def args(self, *arguments):
        return bench.argument_parser().parse_args(arguments)

    def test_small_runs_keep_the_fixed_timeout(self):
        args = self.args()
        self.assertEqual(bench.runner_timeout(args), bench.TIMEOUT_S)
        self.assertEqual(bench.mlx_timeout(args), bench.TIMEOUT_S)

    def test_runs_without_gemm_keep_the_fixed_timeout(self):
        args = self.args("--cases", "sort", "--gemm", "8192", "8192", "8192")
        self.assertEqual(bench.runner_timeout(args), bench.TIMEOUT_S)

    def test_large_runs_allow_their_gemm_work_at_half_a_teraflop(self):
        args = self.args("--gemm", "8192", "8192", "8192", "--samples", "20", "--warmup", "10")
        gemm = 2 * 8192 ** 3
        self.assertEqual(bench.runner_timeout(args), math.ceil(gemm * 64 * 30 * 6 / 0.5e12))
        self.assertEqual(bench.mlx_timeout(args), math.ceil(gemm * 64 * 31 * 2 / 0.5e12))
        self.assertGreater(bench.runner_timeout(args), bench.TIMEOUT_S)

    def test_a_timeout_names_the_shape_and_the_repeat(self):
        error = bench.timed_out("the classic runner", self.args("--gemm", "4096", "512", "8"), 1800)
        self.assertIn("the classic runner did not finish within 1800 s", str(error))
        self.assertIn("--gemm 4096 512 8 --gemm-repeat 64", str(error))

    def test_a_timeout_without_gemm_names_the_cases_instead_of_the_gemm_options(self):
        args = self.args("--cases", "sort,vec-add")
        error = str(bench.timed_out("the metal4 runner", args, 1800))
        self.assertIn("the metal4 runner did not finish within 1800 s for --cases vec-add,sort",
                      error)
        self.assertIn("lower --samples, --warmup or the case sizes", error)
        self.assertNotIn("--gemm", error)

    def test_an_mlx_timeout_becomes_that_error(self):
        def run(args, **options):
            raise subprocess.TimeoutExpired(args, options["timeout"])
        quiet = contextlib.redirect_stdout(io.StringIO())
        with mock.patch.object(bench, "command", run), tempfile.TemporaryDirectory() as temporary:
            with quiet, self.assertRaises(RuntimeError) as failure:
                bench.measure_mlx(Path("python"), self.args(), Path(temporary))
        self.assertIn("MLX did not finish within 1800 s", str(failure.exception))
        self.assertIn("--gemm-repeat 64", str(failure.exception))


class Report(unittest.TestCase):
    REPORT = {"config": {"gemm": [256, 256, 256], "gemm_repeat": 8, "samples": 2},
              "backends": [{"backend": "classic"}], "timing": {"unit": "ns"}}

    def test_the_runner_report_is_written_before_mlx_and_rewritten_with_it(self):
        mlx = mlx_result()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"

            def measure():
                self.assertNotIn("mlx", json.loads(output.read_text()))
                return mlx
            bench.write_report(output, self.REPORT, measure)
            self.assertEqual(json.loads(output.read_text())["mlx"], mlx)

    def test_a_failing_mlx_run_keeps_the_runner_report(self):
        def measure():
            raise RuntimeError("MLX did not finish")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            with self.assertRaises(RuntimeError) as failure:
                bench.write_report(output, self.REPORT, measure)
            self.assertEqual(json.loads(output.read_text()), self.REPORT)
        self.assertIn("MLX did not finish", str(failure.exception))
        self.assertIn("holds the report without MLX", str(failure.exception))

    def test_without_mlx_the_report_is_written_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "report.json"
            self.assertEqual(bench.write_report(output, self.REPORT), self.REPORT)
            self.assertEqual(json.loads(output.read_text()), self.REPORT)


class Timing(unittest.TestCase):
    def test_the_interleave_note_names_the_order_and_the_drift_between_runs(self):
        for words in ("every case", "take turns", "rotating every round", "separate processes",
                      "report the spread", "confirms the interleave"):
            self.assertIn(words, bench.INTERLEAVE_TIMING)


class MlxEnvironment(unittest.TestCase):
    PIN = f"mlx=={bench.MLX_VERSION}"

    def calls(self, *, exists, installed):
        """The commands that mlx_python runs for a venv that has `installed` as its MLX."""
        calls = []

        def run(args, **_):
            calls.append([str(arg) for arg in args])
            if "-c" in args:
                if installed is None:
                    raise RuntimeError("No module named mlx")
                return installed, 0
            return "", 0

        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary) / "venv"
            if exists:
                (venv / "bin").mkdir(parents=True)
                (venv / "bin/python").touch()
            with mock.patch.object(bench, "command", run):
                self.assertEqual(bench.mlx_python(venv), venv / "bin/python")
        return calls

    def test_a_venv_that_has_the_pinned_version_is_reused(self):
        calls = self.calls(exists=True, installed=bench.MLX_VERSION)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("pip", calls[0])

    def test_another_or_missing_mlx_is_replaced_by_the_pin(self):
        for installed in ("0.31.0", None):
            calls = self.calls(exists=True, installed=installed)
            self.assertEqual(len(calls), 2, installed)
            self.assertEqual(calls[1][1:3], ["-m", "pip"])
            self.assertEqual(calls[1][-1], self.PIN)

    def test_a_missing_venv_is_created_first_with_this_interpreter(self):
        calls = self.calls(exists=False, installed=None)
        self.assertEqual(calls[0][:3], [sys.executable, "-m", "venv"])
        self.assertEqual(calls[-1][-1], self.PIN)


class SpectralDriver(unittest.TestCase):
    SOURCES = ("common.metal", "fgn.metal", "sheet.metal", "euler.metal")

    def test_both_modes_are_validated_then_measured_on_both_backends(self):
        builds, hosts = [], []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "examples/spectral/host").mkdir(parents=True)
            for name in self.SOURCES:
                (root / "examples/spectral/host" / name).write_text(name)
            artifact = root / "artifact"
            artifact.mkdir()
            (artifact / "manifest.json").write_text('{"build": {"metal_flags": []}}')

            def build(package, source, environment, log, features):
                builds.append([package, source, *features])
                return artifact, ["cargo"], 1, 2

            def run(args, *, env=None, log=None):
                args = [str(arg) for arg in args]
                if args[0].endswith("/spectral"):
                    backend = env["METAL_OXIDE_BACKEND"]
                    if "--bench" in args:
                        Path(args[2]).write_text(json.dumps({"backend": backend}))
                        args[2] = Path(args[2]).name
                    hosts.append([backend, *args[1:]])
                return ("abc123" if args[:2] == ["git", "rev-parse"] else ""), 0

            output = root / "report.json"
            arguments = ["bench-spectral.py", "--samples", "3", "--warmup", "1", "--max-rows",
                         "16", "--output", str(output)]
            with mock.patch.multiple(spectral, ROOT=root, build_kernels=build, command=run,
                                     digest=lambda path: path.name), \
                    mock.patch.object(sys, "argv", arguments), \
                    contextlib.redirect_stdout(io.StringIO()):
                spectral.main()
            report = json.loads(output.read_text())
        main = "examples/spectral/host/src/main.rs"
        self.assertEqual(builds, [["spectral", main], ["spectral", main, "--features", "fast"]])
        self.assertEqual(hosts, [call for mode in ("precise", "fast")
                                 for backend in ("classic", "metal4")
                                 for call in ([backend], [backend, "--bench",
                                                          f"{mode}-{backend}.json", "3", "1",
                                                          "16"])])
        self.assertEqual([mode["mode"] for mode in report["modes"]], ["precise", "fast"])
        for mode in report["modes"]:
            self.assertEqual(mode["backends"], [{"backend": "classic"}, {"backend": "metal4"}])
            self.assertEqual((mode["first_build_ns"], mode["rebuild_ns"]), (1, 2))
            self.assertEqual(mode["manifest"], {"build": {"metal_flags": []}})
        self.assertEqual((report["revision"], report["dirty"]), ("abc123", False))
        self.assertEqual(report["references"],
                         {f"examples/spectral/host/{name}": name for name in self.SOURCES})


if __name__ == "__main__":
    unittest.main()
