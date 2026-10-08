#!/usr/bin/env python3
"""Interrupt real CLI builds during Metal compilation/linking and verify recovery."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent


def checked(args, *, cwd=ROOT, env=None):
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def main():
    version = checked(["rustc", "-vV"])
    host = next(line.removeprefix("host: ") for line in version.splitlines() if line.startswith("host: "))
    checked(["cargo", "build", "-p", "cargo-metal", "--locked", "--target", host, "--target-dir", ROOT / "target"])
    checked(["cargo", "build", "--features", "rustc-private", "--locked", "--target", host, "--target-dir", ROOT / "target/compiler"], cwd=ROOT / "crates/metal-oxide-compiler")
    component = json.loads(checked(["xcodebuild", "-showComponent", "MetalToolchain", "-json"]))
    directory = Path(component["toolchainSearchPath"]) / "Metal.xctoolchain/usr/metal"
    metal = max((p for p in directory.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)) / "bin/metal"
    with tempfile.TemporaryDirectory(prefix="metal-cache-") as temporary:
        workspace = Path(temporary)
        for name in ["host", "kernels"]:
            (workspace / name / "src").mkdir(parents=True)
        (workspace / "Cargo.toml").write_text('[workspace]\nmembers=["host","kernels"]\nresolver="3"\n')
        (workspace / "host/Cargo.toml").write_text('[package]\nname="host"\nversion="0.1.0"\nedition="2024"\n[package.metadata.metal]\nkernels="../kernels/Cargo.toml"\n')
        (workspace / "host/src/main.rs").write_text('fn main() {}\n')
        (workspace / "kernels/Cargo.toml").write_text('[package]\nname="kernels"\nversion="0.1.0"\nedition="2024"\n[dependencies]\nmetal-oxide-device={path=' + json.dumps(str(ROOT / "crates/metal-oxide-device")) + '}\n')
        (workspace / "kernels/src/lib.rs").write_text('#![no_std]\nuse metal_oxide_device::kernel;\n#[kernel]\npub unsafe fn empty() {}\n')
        checked(["cargo", "generate-lockfile", "--offline"], cwd=workspace)
        environment = dict(os.environ, CARGO_TARGET_DIR=str(workspace / "target"), METAL_OXIDE_COMPILER=str(ROOT / "target/compiler" / host / "debug/metal-oxide-compiler"))
        command = [ROOT / "target" / host / "debug/cargo-metal", "build", "-p", "host"]

        def build():
            log = checked(command, cwd=workspace, env=environment)
            return Path(next(line.split("Metal artifact ", 1)[1] for line in log.splitlines() if "Metal artifact " in line))

        artifact = build()
        root = artifact.parent
        probes = workspace / "probes"
        probes.mkdir()
        shim = probes / "xcrun"
        shim.write_text(f'''#!/usr/bin/env python3
import os,pathlib,sys,time
args=sys.argv[1:]
if 'metal' not in args:
 os.execv({shutil.which("xcrun")!r}, ['xcrun']+args)
args=args[args.index('metal')+1:]
phase='compile' if '-c' in args else 'link'
if '--version' not in args and phase==os.environ['TEST_PHASE']:
 if os.environ['TEST_ACTION']=='fail':
  print('intentional Metal compiler failure',file=sys.stderr)
  sys.exit(7)
 pathlib.Path(os.environ['TEST_MARKER']).write_text('ready')
 parent=os.getppid()
 while os.getppid()==parent: time.sleep(0.05)
 sys.exit(1)
os.execv({str(metal)!r}, [{str(metal)!r}]+args)
''')
        shim.chmod(0o755)
        marker = workspace / "started"
        for phase in ["compile", "link"]:
            (artifact / "kernels.metallib").unlink()
            if marker.exists():
                marker.unlink()
            env = dict(environment, PATH=str(probes) + os.pathsep + os.environ["PATH"], TEST_PHASE=phase, TEST_ACTION="hold", TEST_MARKER=str(marker))
            with (workspace / "interrupted.log").open("w") as log:
                child = subprocess.Popen(command, cwd=workspace, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    deadline = time.monotonic() + 180
                    while not marker.exists():
                        if child.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError((workspace / "interrupted.log").read_text())
                        time.sleep(0.05)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            assert list(root.glob(".build-*")), "interruption did not leave a stage"
            assert build() == artifact
            assert not list(root.glob(".build-*")), "abandoned stage survived recovery"
            print(f"Recovered interrupted Metal {phase}")
        (artifact / "kernels.metallib").unlink()
        env.update(TEST_ACTION="fail", TEST_PHASE="compile")
        failed = subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True, timeout=180)
        assert failed.returncode != 0
        assert "inputs retained at" in failed.stderr, failed.stderr
        stages = list(root.glob(".failed-*"))
        assert len(stages) == 1
        stage = stages[0]
        for filename in ["kernels.metal", "kernels.oxide-ir", "abi.json", "bindings.rs", "metal-commands.json", "metal.log"]:
            assert (stage / filename).is_file(), filename
        assert "intentional Metal compiler failure" in (stage / "metal.log").read_text()
        assert build() == artifact
        assert stage.exists(), "retry discarded failure diagnostics"
        print("Preserved failed Metal inputs and diagnostics; retry succeeds")


if __name__ == "__main__":
    main()
