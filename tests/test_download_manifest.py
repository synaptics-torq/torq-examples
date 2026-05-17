import json
import tempfile
import unittest
from pathlib import Path

from utils.download import read_manifest, verify_manifest, write_manifest


class DownloadManifestTest(unittest.TestCase):
    def test_write_read_and_verify_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            (model_dir / "nested").mkdir(parents=True)
            (model_dir / "a.vmfb").write_text("a")
            (model_dir / "nested" / "b.json").write_text("b")

            manifest_path = write_manifest(
                model_dir,
                "org/repo",
                ["nested/b.json", "a.vmfb"],
            )

            self.assertEqual(manifest_path.name, ".manifest.json")
            manifest = read_manifest(model_dir)
            self.assertIsNotNone(manifest)
            self.assertEqual(manifest["repo_id"], "org/repo")
            self.assertEqual(manifest["files"], ["a.vmfb", "nested/b.json"])
            self.assertTrue(verify_manifest(model_dir))

    def test_verify_manifest_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_manifest(Path(tmp)))
            self.assertFalse(verify_manifest(Path(tmp)))

    def test_verify_manifest_rejects_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / ".manifest.json").write_text("{not json")

            self.assertIsNone(read_manifest(model_dir))
            self.assertFalse(verify_manifest(model_dir))

    def test_verify_manifest_rejects_empty_or_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / ".manifest.json").write_text(
                json.dumps({"repo_id": "org/repo", "files": []})
            )
            self.assertFalse(verify_manifest(model_dir))

            (model_dir / ".manifest.json").write_text(
                json.dumps({"repo_id": "org/repo", "files": ["missing.vmfb"]})
            )
            self.assertFalse(verify_manifest(model_dir))


if __name__ == "__main__":
    unittest.main()
