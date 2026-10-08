#!/usr/bin/env python3
"""Verify a private bundle with an independent application and a stable-only deployment."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def run(args, cwd, env, log, *, success=True):
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    log.write_text(result.stdout)
    if (result.returncode == 0) != success:
        raise RuntimeError(f"unexpected status {result.returncode}: {args!r}\n{result.stdout}")
    print(f"passed: {log.stem}", flush=True)
    return result.stdout


def clean_environment():
    env = dict(os.environ)
    for key in ["RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "METAL_OXIDE_COMPILER",
                "METAL_OXIDE_BINDINGS", "METAL_OXIDE_ARTIFACT_DIR",
                "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"]:
        env.pop(key, None)
    env.update(RUSTUP_TOOLCHAIN="1.99.0", CARGO_NET_OFFLINE="true")
    return env


def stable_deployment(work, source, project, artifact, env):
    deployment = work / "deployment"
    app = deployment / "app"
    shutil.copytree(project / "host", app / "host")
    vendor = deployment / "vendor"
    for name in ["metal-oxide", "metal-oxide-artifact"]:
        shutil.copytree(source / "crates" / name, vendor / "crates" / name)
    cargo = (source / "Cargo.toml").read_text()
    cargo = re.sub(r"members = \[.*?\]", 'members = ["crates/metal-oxide", "crates/metal-oxide-artifact"]', cargo, count=1, flags=re.S)
    cargo = "\n".join(line for line in cargo.splitlines() if "path = " not in line
                      or line.startswith(("metal-oxide =", "metal-oxide-artifact ="))) + "\n"
    (vendor / "Cargo.toml").write_text(cargo)
    cargo = (project / "Cargo.toml").read_text().replace('members = ["helper", "host", "kernels"]', 'members = ["host"]')
    cargo = "\n".join(line for line in cargo.splitlines() if not line.startswith(("external-helper =", "metal-oxide-device ="))) + "\n"
    cargo = re.sub(r'metal-oxide = \{ path = .*? \}', 'metal-oxide = { path = "../vendor/crates/metal-oxide" }', cargo)
    (app / "Cargo.toml").write_text(cargo)
    shutil.copy2(project / "Cargo.lock", app / "Cargo.lock")
    (deployment / "artifact").mkdir()
    for name in ["manifest.json", "kernels.metallib", "bindings.rs"]:
        shutil.copy2(artifact / name, deployment / "artifact" / name)
    env = dict(env, CARGO_TARGET_DIR=str(work / "stable-target"),
               METAL_OXIDE_COMPILER=str(work / "compiler-must-not-run"),
               METAL_OXIDE_BINDINGS=str(deployment / "artifact/bindings.rs"),
               METAL_OXIDE_ARTIFACT_DIR=str(deployment / "artifact"))
    run(["cargo", "generate-lockfile", "--offline"], app, env, work / "stable-lock.log")
    metadata = json.loads(run(["cargo", "metadata", "--format-version", "1", "--locked"], app, env, work / "stable-metadata.log"))
    names = {package["name"] for package in metadata["packages"]}
    assert not names.intersection({"cargo-metal", "metal-oxide-compiler", "metal-oxide-device", "metal-oxide-ir", "metal-oxide-codegen", "metal-oxide-macros"}), names
    output = run(["cargo", "build", "-p", "external-host", "--release", "--locked", "--message-format=json"], app, env, work / "stable-build.log")
    binaries = [message["executable"] for line in output.splitlines() if line.startswith("{")
                for message in [json.loads(line)] if message.get("reason") == "compiler-artifact"
                and message["target"]["name"] == "external-host" and message.get("executable")]
    assert len(binaries) == 1, binaries
    shutil.copy2(binaries[0], deployment / "external-host")
    moved = work / "moved-deployment"
    deployment.rename(moved)
    for backend in ["classic", "metal4"]:
        output = run([moved / "external-host", "--artifact", moved / "artifact", "--n", "1000003"],
                     moved, dict(env, METAL_OXIDE_BACKEND=backend), work / f"moved-{backend}.log")
        assert "n=1000003" in output and "all passes match CPU" in output, output
    manifest = moved / "artifact/manifest.json"
    original = manifest.read_text()
    value = json.loads(original)
    value["abi"]["kernels"][0]["parameters"][0]["name"] += "_stale"
    manifest.write_text(json.dumps(value))
    output = run([moved / "external-host", "--artifact", moved / "artifact"], moved,
                 dict(env, METAL_OXIDE_BACKEND="classic"), work / "stale-bindings.log", success=False)
    assert "ABI" in output, output
    manifest.write_text(original)
    run(["cargo", "clippy", "-p", "external-host", "--all-targets", "--locked", "--", "-D", "warnings"],
        moved / "app", dict(env, METAL_OXIDE_BINDINGS=str(moved / "artifact/bindings.rs"),
                    METAL_OXIDE_ARTIFACT_DIR=str(moved / "artifact")), work / "stable-clippy.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "target").mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="installed-", dir=ROOT / "target"))
    print(f"Acceptance files: {work}", flush=True)
    env = clean_environment()
    prefix = work / "tools"
    run([sys.executable, ROOT / "scripts/package.py", "install", args.archive.resolve(), "--prefix", prefix],
        work, env, work / "install.log")
    source = prefix / "source"
    project = work / "application"
    shutil.copytree(ROOT / "tests/installed", project)
    template = project / "Cargo.toml.in"
    (project / "Cargo.toml").write_text(template.read_text().replace("@CRATES@", str(source / "crates")))
    env.update(CARGO_TARGET_DIR=str(work / "target"), PATH=str(prefix / "bin") + os.pathsep + env["PATH"])
    run(["cargo", "generate-lockfile", "--offline"], project, env, work / "lock.log")
    run(["cargo", "metal", "doctor"], project, env, work / "doctor.log")
    output = run(["cargo", "metal", "inspect", "--manifest-path", project / "host/Cargo.toml", "--emit", "msl"],
                 work, env, work / "inspect.log")
    for name in ["update", "select", "block_scan", "add_offsets", "scatter"]:
        assert f"void {name}(" in output, name
    for backend in ["classic", "metal4"]:
        backend_env = dict(env, METAL_OXIDE_BACKEND=backend)
        for release in [False, True]:
            profile = ["--release"] if release else []
            label = "release" if release else "debug"
            for n in [0, 1, 257, 1_000_003]:
                output = run(["cargo", "metal", "run", "--manifest-path", project / "Cargo.toml",
                              "-p", "external-host"] + profile + ["--", "--n", str(n)], project,
                             backend_env, work / f"{backend}-{label}-{n}.log")
                assert f"n={n}," in output and "all passes match CPU" in output, output
        output = run(["cargo", "metal", "test", "--manifest-path", project / "host/Cargo.toml",
                      "--", "--exact", "installed_chain"], work, backend_env, work / f"test-{backend}.log")
        assert "1 passed" in output, output
    artifacts = list((work / "target/metal").glob("*/manifest.json"))
    assert artifacts, "no compiled artifact"
    artifact = max(artifacts, key=lambda path: path.stat().st_mtime_ns).parent
    stable_deployment(work, source, project, artifact, env)
    print("Installed and relocated stable applications passed on both backends.")


if __name__ == "__main__":
    main()
