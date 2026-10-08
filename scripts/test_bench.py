import importlib.util
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


if __name__ == "__main__":
    unittest.main()
