#!/usr/bin/env python3
"""Verify publishable packages, optionally installing them from a local source mirror."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent.parent


def run(*command, cwd=ROOT, env=None, capture=False):
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def publication_order():
    metadata = json.loads(run(
        "cargo", "metadata", "--format-version", "1", "--no-deps", "--locked",
        capture=True,
    ))
    packages = {
        package["name"]: package for package in metadata["packages"]
        if package["publish"] != []
        and Path(package["manifest_path"]).parent.parent == ROOT / "crates"
    }
    pending = set(packages)
    order = []
    while pending:
        ready = sorted(name for name in pending if not any(
            dependency["name"] in pending and dependency["kind"] != "dev"
            for dependency in packages[name]["dependencies"]
        ))
        if not ready:
            raise RuntimeError(f"publication dependency cycle: {sorted(pending)}")
        for name in ready:
            pending.remove(name)
            order.append(packages[name])
    if not order or len({package["version"] for package in order}) != 1:
        raise RuntimeError("release packages must share one version")
    return order


def mirror(archive, vendor):
    with tarfile.open(archive) as source:
        directory = vendor / source.getnames()[0].split("/")[0]
        source.extractall(vendor, filter="data")
    files = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }
    (directory / ".cargo-checksum.json").write_text(json.dumps({
        "files": files,
        "package": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }))


def smoke(packages, target):
    with tempfile.TemporaryDirectory(prefix="metal-oxide-release-") as temporary:
        directory = Path(temporary).resolve()
        vendor = directory / "vendor"
        run("cargo", "vendor", "--quiet", "--locked", "--versioned-dirs", str(vendor), capture=True)
        version = packages[0]["version"]
        for package in packages:
            mirror(target / "package" / f"{package['name']}-{version}.crate", vendor)
        cargo_home = directory / "cargo"
        cargo_home.mkdir()
        (cargo_home / "config.toml").write_text(
            '[source.crates-io]\nreplace-with = "release"\n'
            f'[source.release]\ndirectory = {json.dumps(str(vendor))}\n'
        )
        tools = directory / "tools"
        environment = dict(os.environ, CARGO_HOME=str(cargo_home))
        for name in ("CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "METAL_OXIDE_COMPILER",
                     "METAL_OXIDE_ARTIFACTS", "METAL_OXIDE_CAPTURE"):
            environment.pop(name, None)
        environment["PATH"] = str(tools / "bin") + os.pathsep + environment["PATH"]
        run("cargo", "install", "cargo-metal", "--version", f"={version}",
            "--locked", "--root", str(tools), env=environment)
        run(str(tools / "bin/cargo-metal"), "metal", "setup", env=environment)
        toolchain = tomllib.loads((ROOT / "crates/metal-oxide-compiler/rust-toolchain.toml").read_text())
        nightly = toolchain["toolchain"]["channel"]
        sysroot = run("rustup", "run", nightly, "rustc", "--print", "sysroot", capture=True).strip()
        run("rustup", "run", nightly, "cargo", "vendor", "--quiet", "--locked",
            "--no-delete", "--versioned-dirs", "--sync",
            str(Path(sysroot) / "lib/rustlib/src/rust/library/Cargo.toml"),
            str(vendor), capture=True)
        application = directory / "application"
        shutil.copytree(ROOT / "tests/installed", application)
        manifest = application / "Cargo.toml"
        text = manifest.read_text().replace(
            'metal-oxide = { path = "../../crates/metal-oxide" }',
            f'metal-oxide = "={version}"',
        ).replace('installed-helper = { path = "helper" }',
                  'installed-helper = { path = "helper", version = "=0.0.0" }')
        manifest.write_text(text)
        run("cargo", "run", "--offline", cwd=application, env=environment)
        for name in ("installed-helper", "installed"):
            run("cargo", "package", "-p", name, "--locked", "--offline",
                "--registry", "crates-io", cwd=application, env=environment)
            mirror(application / "target/package" / f"{name}-0.0.0.crate", vendor)
        environment["METAL_OXIDE_COMPILER"] = str(directory / "missing-compiler")
        environment["METAL_OXIDE_ARTIFACTS"] = "locked"
        run("cargo", "+1.89.0", "install", "installed", "--version", "=0.0.0",
            "--locked", "--offline", "--root", str(tools), env=environment)
        run(str(tools / "bin/installed"), env=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="install the packaged tools and run kernels on this Mac")
    args = parser.parse_args()
    packages = publication_order()
    print("Publication order: " + " -> ".join(p["name"] for p in packages), flush=True)
    target = ROOT / "target/release-check"
    command = ["cargo", "package", "--locked", "--target-dir", str(target)]
    if args.allow_dirty:
        command.append("--allow-dirty")
    for package in packages:
        command.extend(["-p", package["name"]])
    with tempfile.TemporaryDirectory(prefix="metal-oxide-packaging-") as cargo_home:
        run(*command, env=dict(os.environ, CARGO_HOME=cargo_home))
    if args.smoke:
        smoke(packages, target)
    print(f"Verified {len(packages)} packages" + (" and installation smoke tests." if args.smoke else "."))


if __name__ == "__main__":
    main()
