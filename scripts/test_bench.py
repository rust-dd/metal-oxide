import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


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


if __name__ == "__main__":
    unittest.main()
