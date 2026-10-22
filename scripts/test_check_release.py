import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location("check_release",
                                              Path(__file__).with_name("check-release.py"))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class Precedence(unittest.TestCase):
    def test_orders_dotted_prereleases_numerically_and_before_their_release(self):
        ordered = ["0.1.0-alpha.1", "0.1.0-alpha.2", "0.1.0-alpha.10", "0.1.0-alpha0",
                   "0.1.0-beta.1", "0.1.0", "0.2.0-alpha.1"]
        self.assertEqual(sorted(reversed(ordered), key=release.precedence), ordered)

    def test_numeric_identifiers_sort_below_alphanumeric_ones(self):
        self.assertLess(release.precedence("1.0.0-alpha.1"),
                        release.precedence("1.0.0-alpha.beta"))


class Version(unittest.TestCase):
    def check(self, version, published, supersedes=()):
        with mock.patch.object(release, "published", return_value=published):
            release.check_version(version, ["metal-oxide"], supersedes)

    def test_accepts_a_version_above_every_live_one(self):
        self.check("0.1.0-alpha.2", [("0.1.0-alpha.1", False)])

    def test_rejects_a_version_that_was_ever_published(self):
        with self.assertRaisesRegex(RuntimeError, "already published"):
            self.check("0.1.0-alpha.1", [("0.1.0-alpha.1", True)])

    def test_rejects_a_version_below_a_live_one(self):
        with self.assertRaisesRegex(RuntimeError, "sorts below published 0.1.0-alpha0"):
            self.check("0.1.0-alpha.1", [("0.1.0-alpha0", False)])

    def test_ignores_yanked_and_superseded_versions(self):
        self.check("0.1.0-alpha.1", [("0.1.0-alpha0", True)])
        self.check("0.1.0-alpha.1", [("0.1.0-alpha0", False)], supersedes=["0.1.0-alpha0"])

    def test_index_paths_follow_the_crates_io_layout(self):
        self.assertEqual(release.index_path("metal-oxide"), "me/ta/metal-oxide")
        self.assertEqual(release.index_path("Abc"), "3/a/abc")
        self.assertEqual(release.index_path("ab"), "2/ab")


class Pins(unittest.TestCase):
    INSTALL = "cargo install cargo-metal --version '={0}' --locked\ncargo add metal-oxide@={0}\n"

    def pins(self, version, readme, installation):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "book/src").mkdir(parents=True)
            (root / "README.md").write_text(readme)
            (root / "book/src/installation.md").write_text(installation)
            release.check_pins(version, root)

    def test_accepts_install_commands_and_tags_of_the_version(self):
        version = "0.1.0-alpha.1"
        self.pins(version, self.INSTALL.format(version) + f"git clone --branch v{version} url\n",
                  self.INSTALL.format(version) + f"check out `v{version}` before\n")

    def test_reports_each_stale_pin_with_its_line(self):
        readme = "cargo install cargo-metal --version '=0.1.0-alpha.1' --locked\n" \
                 "cargo add metal-oxide@=0.1.0-alpha0\n"
        with self.assertRaisesRegex(RuntimeError, "README.md:2: 0.1.0-alpha0"):
            self.pins("0.1.0-alpha.1", readme, self.INSTALL.format("0.1.0-alpha.1"))

    def test_requires_each_file_to_name_a_version(self):
        with self.assertRaisesRegex(RuntimeError, "installation.md names no release version"):
            self.pins("0.1.0-alpha.1", self.INSTALL.format("0.1.0-alpha.1"), "no commands\n")


class Checks(unittest.TestCase):
    def check(self, runs):
        response = json.dumps({"total_count": len(runs), "check_runs": runs})
        with mock.patch.object(release, "fetch", return_value=response):
            release.check_ci("rust-dd/metal-oxide", "abc")

    def test_accepts_completed_successful_and_skipped_runs(self):
        self.check([{"name": "CI", "status": "completed", "conclusion": "success"},
                    {"name": "registry", "status": "completed", "conclusion": "skipped"}])

    def test_rejects_missing_unfinished_and_failed_runs(self):
        for runs, message in [
            ([], "no checks"),
            ([{"name": "CI", "status": "in_progress", "conclusion": None}], "CI: in_progress"),
            ([{"name": "CI", "status": "completed", "conclusion": "failure"}], "CI: failure"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.check(runs)

    def test_reports_a_commit_that_github_does_not_have(self):
        missing = release.urllib.error.HTTPError("url", 422, "Unprocessable Entity", None, None)
        with mock.patch.object(release, "fetch", side_effect=missing), \
                self.assertRaisesRegex(RuntimeError, "GitHub has no commit abc; push it"):
            release.check_ci("rust-dd/metal-oxide", "abc")


if __name__ == "__main__":
    unittest.main()
