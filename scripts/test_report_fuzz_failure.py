import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
import zipfile


sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "report_fuzz_failure", Path(__file__).with_name("report-fuzz-failure.py")
)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)

RUN = {
    "id": 123,
    "head_sha": "a" * 40,
    "html_url": "https://github.com/example/metal-oxide/actions/runs/123",
}
SEED = "2c83c99044574110"
ERROR = "the device fails: Metal pipeline error: Compilation failed"
SOURCE = "fn cooperate() -> [u32; 1] { [0_u32] }\n"


def job(os_name="macos-15", conclusion="failure", step="Compare generated programs"):
    return {
        "id": 456 if os_name == "macos-15" else 789,
        "name": f"Generated programs ({os_name})",
        "conclusion": conclusion,
        "html_url": f"{RUN['html_url']}/job/{456 if os_name == 'macos-15' else 789}",
        "steps": [{"name": step, "conclusion": conclusion}],
    }


def archive(files=None):
    if files is None:
        files = {f"{SEED}/failure.txt": ERROR, f"{SEED}/minimized.rs": SOURCE}
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as zipped:
        for path, contents in files.items():
            zipped.writestr(path, contents)
    return data.getvalue()


class FakeGitHub(reporter.GitHub):
    def __init__(self):
        super().__init__("example/metal-oxide")
        self.attempts = {1: [job(), job("macos-26", "success")]}
        self.artifacts = [{"id": 42, "name": "fuzz-macos-15-attempt-1"}]
        self.issues = []
        self.downloads = {42: archive()}
        self.logs = b"comparison failed\n"
        self.writes = []

    def request(self, path, *, method="GET", payload=None, paginate=False):
        if method in {"POST", "PATCH"}:
            self.writes.append((path, method, copy.deepcopy(payload)))
            if method == "POST":
                number = len(self.issues) + 1
                issue = {
                    "number": number,
                    "html_url": f"https://github.com/example/metal-oxide/issues/{number}",
                    "state": "open",
                    **payload,
                }
                self.issues.append(issue)
            else:
                issue = next(item for item in self.issues if path == f"issues/{item['number']}")
                issue.update(payload)
            return json.dumps(issue).encode()
        if path == "actions/runs/123":
            value = RUN
        elif "/attempts/" in path:
            attempt = int(path.split("/attempts/")[1].split("/")[0])
            value = {"jobs": self.attempts[attempt]}
        elif path == "actions/runs/123/artifacts?per_page=100":
            value = {"artifacts": self.artifacts}
        elif path == "issues?state=all&per_page=100":
            value = self.issues
        elif path.startswith("actions/artifacts/"):
            result = self.downloads[int(path.split("/")[2])]
            if isinstance(result, Exception):
                raise result
            return result
        elif path.startswith("actions/jobs/") and path.endswith("/logs"):
            if isinstance(self.logs, Exception):
                raise self.logs
            return self.logs
        else:
            raise AssertionError(f"unexpected API path: {path}")
        return json.dumps([value] if paginate else value).encode()


class Reports(unittest.TestCase):
    def setUp(self):
        self.api = FakeGitHub()

    def report(self, attempt=1, *, dry_run=False):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            reporter.report(self.api, 123, attempt, "classic", dry_run=dry_run)
        return output.getvalue()

    def test_failure_includes_seed_reproduction_source_and_links(self):
        self.report()
        self.assertEqual(len(self.api.writes), 1)
        path, method, payload = self.api.writes[0]
        self.assertEqual((path, method), ("issues", "POST"))
        self.assertIn("macos-15 (classic, run 123)", payload["title"])
        body = payload["body"]
        for expected in (
            ERROR, f"0x{SEED}", "3207628981076967696", SOURCE.strip(),
            "METAL_OXIDE_FUZZ_SEED=3207628981076967696",
            "METAL_OXIDE_FUZZ_SECONDS=1", "METAL_OXIDE_FUZZ_SHRINK_SECONDS=0",
            "--profile fuzz --test fuzz --run-ignored only --no-capture",
            f"{RUN['html_url']}/attempts/1", f"{RUN['html_url']}/job/456",
            f"{RUN['html_url']}/artifacts/42", f"/commit/{RUN['head_sha']}",
            "comparison failed", "<!-- metal-oxide-fuzz:123:macos-15 -->",
        ):
            self.assertIn(expected, body)
        self.assertNotIn("macos-26", body)

    def test_each_failed_runner_gets_its_own_issue(self):
        self.api.attempts[1][1] = job("macos-26")
        self.report()
        self.assertEqual(len(self.api.writes), 2)
        self.assertIn("macos-26", self.api.writes[1][2]["title"])
        self.assertIn("No saved failing seed", self.api.writes[1][2]["body"])

    def test_rerun_updates_same_issue_even_when_closed(self):
        self.report()
        self.api.issues[0]["state"] = "closed"
        self.api.attempts[2] = [job()]
        self.api.artifacts.append({"id": 43, "name": "fuzz-macos-15-attempt-2"})
        self.api.downloads[43] = archive({
            "0000000000000001/failure.txt": "new failure",
        })
        self.report(2)
        self.assertEqual(len(self.api.issues), 1)
        self.assertEqual(self.api.writes[-1][:2], ("issues/1", "PATCH"))
        self.assertEqual(self.api.issues[0]["state"], "closed")
        body = self.api.writes[-1][2]["body"]
        self.assertIn("new failure", body)
        self.assertIn("/attempts/2", body)
        self.assertIn("/artifacts/43", body)
        self.assertNotIn(ERROR, body)

    def test_rerunning_only_reporter_uses_failed_jobs_original_attempt(self):
        self.api.attempts[2] = []
        self.report(2)
        body = self.api.writes[0][2]["body"]
        self.assertIn(ERROR, body)
        self.assertIn("/attempts/1", body)
        self.assertIn("/artifacts/42", body)

    def test_successful_rerun_does_not_report_an_old_failure(self):
        self.api.attempts[2] = [job(conclusion="success")]
        self.assertIn("No failed fuzz matrix jobs", self.report(2))
        self.assertEqual(self.api.writes, [])

    def test_no_artifact_reports_retained_regression_log_and_command(self):
        self.api.artifacts = []
        self.api.attempts[1][0] = job(step="Compare retained regressions on classic Metal")
        self.api.logs = b"retained_reverse_float failed: 22351.0 != 798.25\n"
        self.report()
        body = self.api.writes[0][2]["body"]
        self.assertIn("No saved-program artifact", body)
        self.assertIn("No saved failing seed", body)
        self.assertIn("retained_reverse_float failed: 22351.0 != 798.25", body)
        self.assertIn("--profile gpu --test differential --run-ignored only", body)

    def test_setup_failure_reports_step_and_log_without_inventing_a_seed(self):
        self.api.artifacts = []
        self.api.attempts[1][0] = job(step="Install compiler and workspace toolchains")
        self.api.logs = b"error: component download failed"
        self.report()
        body = self.api.writes[0][2]["body"]
        self.assertIn("Install compiler and workspace toolchains", body)
        self.assertIn("error: component download failed", body)
        self.assertNotIn("METAL_OXIDE_FUZZ_SEED=", body)

    def test_cleanup_output_cannot_displace_the_failure_log(self):
        self.api.artifacts = []
        self.api.logs = (
            b"retained regression failed: wrong result\n"
            b"##[error]Process completed with exit code 100.\n"
            + b"Post job cleanup: removing credentials configuration\n" * 1000
        )
        self.report()
        body = self.api.writes[0][2]["body"]
        self.assertIn("retained regression failed: wrong result", body)
        self.assertNotIn("Post job cleanup", body)

    def test_incomplete_minimization_keeps_seed_and_failure(self):
        self.api.downloads[42] = archive({f"{SEED}/failure.txt": ERROR})
        self.report()
        body = self.api.writes[0][2]["body"]
        self.assertIn(ERROR, body)
        self.assertIn("Minimization did not produce a saved source file", body)

    def test_unavailable_artifact_and_logs_still_create_a_linked_issue(self):
        self.api.downloads[42] = subprocess.CalledProcessError(1, ["gh", "api"])
        self.api.logs = subprocess.CalledProcessError(1, ["gh", "api"])
        self.report()
        body = self.api.writes[0][2]["body"]
        self.assertIn("Could not read the saved programs", body)
        self.assertIn("Job logs are not available", body)
        self.assertIn("/job/456", body)

    def test_corrupt_artifact_does_not_hide_the_job_failure(self):
        self.api.downloads[42] = b"not a ZIP"
        self.report()
        body = self.api.writes[0][2]["body"]
        self.assertIn("BadZipFile", body)
        self.assertIn("comparison failed", body)

    def test_dry_run_prints_report_without_mutation(self):
        output = self.report(dry_run=True)
        self.assertIn(ERROR, output)
        self.assertEqual(self.api.writes, [])

    def test_source_and_log_excerpts_keep_issue_within_github_limit(self):
        self.api.downloads[42] = archive({
            f"{seed:016x}/{name}": text
            for seed in range(4)
            for name, text in (("failure.txt", "e" * 100_000),
                               ("minimized.rs", "fn example() {}\n" * 20_000))
        })
        self.api.logs = b"irrelevant\n" * 10_000 + b"final error\n"
        self.report()
        body = self.api.writes[0][2]["body"]
        self.assertLess(len(body), 60_000)
        self.assertIn("1 additional failing seeds", body)
        self.assertIn("final error", body)
        self.assertIn("[earlier output omitted]", body)

    def test_unrelated_pull_requests_and_run_markers_do_not_suppress_issue(self):
        marker = "<!-- metal-oxide-fuzz:123:macos-15 -->"
        self.api.issues = [
            {"number": 1, "pull_request": {}, "body": marker},
            {"number": 2, "body": "<!-- metal-oxide-fuzz:122:macos-15 -->"},
        ]
        self.report()
        self.assertEqual(self.api.writes[-1][:2], ("issues", "POST"))


class DataHandling(unittest.TestCase):
    def test_zip_paths_are_read_as_data_without_extraction(self):
        content = archive({
            "../outside/failure.txt": "not a seed",
            f"{SEED}/failure.txt": "$(touch /tmp/should-not-exist)",
            f"{SEED}/minimized.rs": "```\n# outside fence?\n```",
        })
        with mock.patch.object(zipfile.ZipFile, "extractall") as extract:
            failures = reporter.saved_failures(content)
        extract.assert_not_called()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][1], "$(touch /tmp/should-not-exist)")
        fenced = reporter.code(failures[0][2], "rust")
        self.assertTrue(fenced.startswith("````rust\n"))
        self.assertTrue(fenced.endswith("\n````"))

    def test_issue_payload_is_json_on_stdin_not_shell_or_cli_fields(self):
        api = reporter.GitHub("example/metal-oxide")
        payload = {"body": "first line\n$(touch /tmp/should-not-exist)\n`example`"}
        with mock.patch.object(subprocess, "run") as run:
            run.return_value.stdout = b'{"number":1}'
            api.json("issues", method="POST", payload=payload)
        command = run.call_args.args[0]
        self.assertEqual(command, [
            "gh", "api", "repos/example/metal-oxide/issues",
            "--method", "POST", "--input", "-",
        ])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), payload)

    def test_all_paginated_issues_are_considered(self):
        api = reporter.GitHub("example/metal-oxide")
        with mock.patch.object(subprocess, "run") as run:
            run.return_value.stdout = b'[[{"number":1}],[{"number":2}]]'
            issues = api.list("issues?state=all&per_page=100")
        self.assertEqual(issues, [{"number": 1}, {"number": 2}])
        self.assertEqual(run.call_args.args[0][-2:], ["--paginate", "--slurp"])


if __name__ == "__main__":
    unittest.main()
