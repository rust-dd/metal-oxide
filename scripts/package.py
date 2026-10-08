#!/usr/bin/env python3
"""Create or install a private macOS alpha bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent.parent


def run(args, cwd=ROOT):
    return subprocess.check_output(list(map(str, args)), cwd=cwd, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def licenses(bundle, toolchain):
    metadata = json.loads(run(["cargo", "metadata", "--format-version", "1", "--locked"]))
    for package in metadata["packages"]:
        if package["source"] is None:
            continue
        directory = Path(package["manifest_path"]).parent
        files = [p for p in directory.iterdir() if p.is_file()
                 and p.name.lower().startswith(("license", "copying", "notice", "copyright"))]
        if package.get("license_file"):
            files.append(directory / package["license_file"])
        if not files and package["repository"] == "https://github.com/madsmtm/objc2":
            files.append(ROOT / "licenses/objc2.md")
        if not files:
            raise ValueError(f"missing license text for {package['name']}")
        destination = bundle / "licenses" / f"{package['name']}-{package['version']}"
        destination.mkdir(parents=True)
        (destination / "package.json").write_text(json.dumps({key: package[key] for key in ["name", "version", "license", "repository"]}, indent=2) + "\n")
        for path in files:
            shutil.copy2(path, destination / path.name)
    for name, prefix in [("host-rust", ["rustc"]), ("compiler-rust", ["rustup", "run", toolchain, "rustc"])]:
        sysroot = Path(run(prefix + ["--print", "sysroot"]))
        destination = bundle / "licenses" / name
        destination.mkdir(parents=True)
        for filename in ["COPYRIGHT.html", "COPYRIGHT-library.html"]:
            shutil.copy2(sysroot / "share/doc/rust" / filename, destination / filename)


def build(output):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("private binary bundles require macOS on Apple Silicon")
    if run(["git", "status", "--porcelain"]):
        raise ValueError("commit source changes before packaging a release")
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
    version = cargo["workspace"]["package"]["version"]
    toolchain = tomllib.loads((ROOT / "crates/metal-oxide-compiler/rust-toolchain.toml").read_text())["toolchain"]["channel"]
    name = f"metal-oxide-{version}-aarch64-apple-darwin"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}.tar.gz"
    if archive.exists():
        raise ValueError(f"refusing to replace {archive}; use a fresh output directory")
    target = ROOT / "target/package-build"
    for package, prefix, features in [
        ("cargo-metal", ["cargo"], []),
        ("metal-oxide-compiler", ["rustup", "run", toolchain, "cargo"], ["--features", "rustc-private"]),
    ]:
        run(prefix + ["build", "-p", package, "--bin", package, "--release", "--locked",
                      "--target", "aarch64-apple-darwin", "--target-dir", target] + features)
    with tempfile.TemporaryDirectory(prefix="bundle-", dir=output) as temporary:
        bundle = Path(temporary) / name
        source = bundle / "source"
        source.mkdir(parents=True)
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        for relative in filter(None, tracked):
            if relative.startswith((".agents/", ".github/")) or relative == "AGENTS.md":
                continue
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        (bundle / "bin").mkdir()
        for binary in ["cargo-metal", "metal-oxide-compiler"]:
            shutil.copy2(target / "aarch64-apple-darwin/release" / binary, bundle / "bin" / binary)
        shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
        licenses(bundle, toolchain)
        (bundle / "release.json").write_text(json.dumps({"version": version, "revision": run(["git", "rev-parse", "HEAD"]), "nightly": toolchain, "host": "aarch64-apple-darwin"}, indent=2) + "\n")
        hashes = {str(p.relative_to(bundle)): digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
        (bundle / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(bundle, arcname=name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest(archive)}  {archive.name}\n")
    print(archive)


def install(archive, prefix):
    if prefix.exists():
        raise ValueError(f"installation path already exists: {prefix}; choose a fresh versioned prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".metal-install-", dir=prefix.parent) as temporary:
        stage = Path(temporary)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if any(not (m.isfile() or m.isdir()) or Path(m.name).is_absolute() or ".." in Path(m.name).parts for m in members):
                raise ValueError("bundle contains unsupported archive entries")
            # Links and paths escaping stage were rejected before extraction.
            tar.extractall(stage, members=members)
        roots = list(stage.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("bundle must contain one root directory")
        bundle = roots[0]
        hashes = json.loads((bundle / "checksums.json").read_text())
        actual = {str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*") if p.is_file() and p != bundle / "checksums.json"}
        if hashes != actual:
            raise ValueError("bundle checksum mismatch")
        release = json.loads((bundle / "release.json").read_text())
        if platform.system() != "Darwin" or platform.machine() != "arm64" or release["host"] != "aarch64-apple-darwin":
            raise ValueError("bundle requires macOS on Apple Silicon")
        subprocess.run([bundle / "bin/cargo-metal", "doctor"], check=True)
        bundle.rename(prefix)
    print(f"Installed {release['version']} in {prefix}; add {prefix / 'bin'} to PATH")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("build")
    create.add_argument("--output", type=Path, default=ROOT / "target/dist")
    setup = commands.add_parser("install")
    setup.add_argument("archive", type=Path)
    setup.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "build":
        build(args.output.resolve())
    else:
        install(args.archive.resolve(), args.prefix.resolve())


if __name__ == "__main__":
    main()
