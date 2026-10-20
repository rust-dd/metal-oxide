#!/usr/bin/env python3
"""Report failed Fuzz matrix jobs using their saved programs and job logs."""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import zipfile


NEXTEST = (
    "cargo nextest run --features rustc-private --locked --target-dir ../../target/compiler"
)
FAILURES = {"failure", "timed_out", "startup_failure"}


class GitHub:
    def __init__(self, repository):
        self.repository = repository

    def request(self, path, *, method="GET", payload=None, paginate=False):
        command = ["gh", "api", f"repos/{self.repository}/{path}", "--method", method]
        if paginate:
            command.extend(["--paginate", "--slurp"])
        if payload is not None:
            command.extend(["--input", "-"])
        result = subprocess.run(
            command,
            input=json.dumps(payload).encode() if payload is not None else None,
            capture_output=True,
            check=True,
        )
        return result.stdout

    def json(self, path, **options):
        return json.loads(self.request(path, **options))

    def list(self, path, key=None):
        pages = self.json(path, paginate=True)
        return [item for page in pages for item in (page[key] if key else page)]


def excerpt(text, limit, *, tail=False):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).strip()
    if len(text) <= limit:
        return text
    if tail:
        return "[earlier output omitted]\n" + text[-limit:]
    return text[:limit] + "\n[remaining output omitted]"


def code(text, language="text"):
    longest = max((len(match[0]) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def failure_log(logs):
    error = logs.rfind("##[error]")
    if error >= 0:
        end = logs.find("\n", error)
        logs = logs[:end] if end >= 0 else logs
    return excerpt(logs, 12000, tail=True)


def saved_failures(archive):
    failures = []
    with zipfile.ZipFile(io.BytesIO(archive)) as files:
        paths = set(files.namelist())
        for path in sorted(paths):
            match = re.fullmatch(r"([0-9a-f]{16})/failure\.txt", path)
            if not match:
                continue
            seed = match[1]
            with files.open(path) as source:
                failure = source.read(4001).decode("utf-8", errors="replace")
            minimized = f"{seed}/minimized.rs"
            program = None
            if minimized in paths:
                with files.open(minimized) as source:
                    program = source.read(6001).decode("utf-8", errors="replace")
            failures.append((seed, failure, program))
    return failures


def reproduction(seed, backend):
    return (
        "cd crates/metal-oxide-compiler\n"
        f"METAL_OXIDE_BACKEND={backend} METAL_OXIDE_FUZZ_GPU=1 \\\n"
        f"  METAL_OXIDE_FUZZ_SEED={int(seed, 16)} METAL_OXIDE_FUZZ_SECONDS=1 \\\n"
        "  METAL_OXIDE_FUZZ_SHRINK_SECONDS=0 \\\n"
        f"  {NEXTEST} \\\n"
        "  --profile fuzz --test fuzz --run-ignored only --no-capture"
    )


def issue_body(run, job, os_name, backend, attempt, artifact, failures, logs, notes):
    run_url = run["html_url"]
    repository_url = run_url.split("/actions/runs/")[0]
    marker = f"<!-- metal-oxide-fuzz:{run['id']}:{os_name} -->"
    failed_steps = [
        step["name"] for step in job.get("steps", []) if step.get("conclusion") in FAILURES
    ]
    parts = [
        marker,
        f"Fuzz CI failed on `{os_name}` with backend `{backend}`.",
        f"- Commit: [{run['head_sha']}]({repository_url}/commit/{run['head_sha']})",
        f"- Run: [attempt {attempt}]({run_url}/attempts/{attempt})",
        f"- Job: [{job['name']}]({job['html_url']})",
        f"- Result: `{job['conclusion']}`",
    ]
    if failed_steps:
        parts.append("- Failed steps: " + ", ".join(f"`{step}`" for step in failed_steps))
    if artifact:
        artifact_url = f"{run_url}/artifacts/{artifact['id']}"
        parts.append(f"- Full programs: [{artifact['name']}]({artifact_url})")
    else:
        parts.append("- No saved-program artifact was uploaded for this attempt.")
    parts.append(
        "\nThe commands below run from the repository root at the linked commit, "
        "on an Apple Silicon Mac with the repository toolchains and nextest installed."
    )
    for seed, failure, program in failures[:3]:
        parts.extend([
            f"\n### Seed `0x{seed}` (decimal `{int(seed, 16)}`)",
            code(excerpt(failure, 4000)),
            "\nReplay the failing seed with a one-second generation budget:",
            code(reproduction(seed, backend), "sh"),
        ])
        if program is not None:
            lines = program.splitlines()
            snippet = "\n".join(lines[:80])
            if len(lines) > 80:
                snippet += "\n[remaining lines omitted]"
            parts.extend([
                "\nMinimized source excerpt; the artifact contains the full file:",
                code(excerpt(snippet, 6000), "rust"),
            ])
        else:
            parts.append("\nMinimization did not produce a saved source file.")
    if len(failures) > 3:
        parts.append(f"\nThe artifact contains {len(failures) - 3} additional failing seeds.")
    if not failures:
        parts.append(
            "\nNo saved failing seed was available. The failed steps and log excerpt "
            "below cover setup, retained-regression, and interrupted fuzz failures."
        )
        if any("retained regressions" in step for step in failed_steps):
            parts.extend([
                "\nReplay the retained regressions:",
                code(
                    "cd crates/metal-oxide-compiler\n"
                    f"METAL_OXIDE_BACKEND={backend} {NEXTEST} \\\n"
                    "  --profile gpu --test differential --run-ignored only --no-capture",
                    "sh",
                ),
            ])
    if notes:
        parts.extend(["\n### Report limitations", *notes])
    parts.extend(["\n### Job log excerpt", code(failure_log(logs))])
    return "\n\n".join(parts) + "\n"


def failed_matrix_jobs(api, run_id, attempt):
    latest = {}
    for number in range(attempt, 0, -1):
        jobs = api.list(f"actions/runs/{run_id}/attempts/{number}/jobs?per_page=100", "jobs")
        for job in jobs:
            if re.fullmatch(r"Generated programs \(macos-\d+\)", job["name"]):
                latest.setdefault(job["name"], (job, number))
    return [item for item in latest.values() if item[0].get("conclusion") in FAILURES]


def report(api, run_id, attempt, backend, *, dry_run=False):
    run = api.json(f"actions/runs/{run_id}")
    failed_jobs = failed_matrix_jobs(api, run_id, attempt)
    if not failed_jobs:
        print("No failed fuzz matrix jobs.")
        return
    artifacts = api.list(f"actions/runs/{run_id}/artifacts?per_page=100", "artifacts")
    issues = api.list("issues?state=all&per_page=100")
    for job, job_attempt in failed_jobs:
        os_name = job["name"].removeprefix("Generated programs (").removesuffix(")")
        artifact_name = f"fuzz-{os_name}-attempt-{job_attempt}"
        artifact = next((item for item in artifacts if item["name"] == artifact_name), None)
        failures, notes = [], []
        if artifact:
            try:
                failures = saved_failures(api.request(f"actions/artifacts/{artifact['id']}/zip"))
            except (subprocess.CalledProcessError, zipfile.BadZipFile, OSError) as error:
                notes.append(f"Could not read the saved programs: `{type(error).__name__}`.")
        try:
            logs = api.request(f"actions/jobs/{job['id']}/logs").decode("utf-8", errors="replace")
        except subprocess.CalledProcessError:
            logs = "Job logs are not available through the API. Follow the job link above."
        marker = f"<!-- metal-oxide-fuzz:{run_id}:{os_name} -->"
        existing = next((
            issue for issue in issues
            if "pull_request" not in issue and marker in (issue.get("body") or "")
        ), None)
        payload = {
            "title": f"Fuzz failure on {os_name} ({backend}, run {run_id})",
            "body": issue_body(
                run, job, os_name, backend, job_attempt, artifact, failures, logs, notes
            ),
        }
        if dry_run:
            print(payload["title"] + "\n\n" + payload["body"])
        elif existing:
            updated = api.json(f"issues/{existing['number']}", method="PATCH", payload=payload)
            print(f"Updated {updated['html_url']}")
        else:
            created = api.json("issues", method="POST", payload=payload)
            issues.append(created)
            print(f"Created {created['html_url']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print issues without posting")
    args = parser.parse_args()
    report(
        GitHub(os.environ["GITHUB_REPOSITORY"]),
        int(os.environ["GITHUB_RUN_ID"]),
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
        os.environ["METAL_OXIDE_BACKEND"],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
