import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("package", Path(__file__).with_name("package.py"))
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class Installation(unittest.TestCase):
    def test_existing_prefix_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            sentinel = prefix / "keep"
            sentinel.write_text("existing installation")
            with self.assertRaisesRegex(ValueError, "already exists"):
                package.install(prefix / "missing.tar.gz", prefix)
            self.assertEqual(sentinel.read_text(), "existing installation")

    def test_corruption_and_escaping_archive_entries_are_rejected(self):
        for name in ["bundle/value", "../escape", "/absolute"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "bundle.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    for path, data in [(name, b"corrupt"), ("bundle/checksums.json", json.dumps({"value": "wrong"}).encode())]:
                        entry = tarfile.TarInfo(path)
                        entry.size = len(data)
                        tar.addfile(entry, io.BytesIO(data))
                with self.assertRaisesRegex(ValueError, "checksum mismatch|unsupported archive"):
                    package.install(archive, root / "installed")
                self.assertFalse((root / "installed").exists())
                self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
