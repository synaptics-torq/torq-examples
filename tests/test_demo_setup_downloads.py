import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gemma3 import setup_demo as gemma_setup
from moonshine import setup_demo as moonshine_setup
from utils.download import write_manifest

_REVISION = "abc123"


def _fake_download(default_base_dir: Path):
    def download(repo_id: str, filename: str, *, base_dir: Path | None = None):
        root = Path(base_dir) if base_dir is not None else default_base_dir
        path = root / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
        return path

    return download


class DemoSetupDownloadsTest(unittest.TestCase):
    def test_gemma_skips_when_revision_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = gemma_setup._HF_REPO_MAP["instruct"]
            model_dir = base_dir / repo_id
            files = [
                "model.vmfb.trim",
                *gemma_setup._GEMMA3_REQUIRED_FILES,
            ]
            for filename in files:
                path = model_dir / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(filename)
            write_manifest(model_dir, repo_id, files, revision=_REVISION)

            with (
                mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(gemma_setup, "check_requirements"),
                mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
                mock.patch.object(gemma_setup, "download_from_hf") as download,
            ):
                gemma_setup.setup_gemma3(["instruct"])

            download.assert_not_called()

    def test_gemma_repairs_incomplete_download_and_records_lut(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = gemma_setup._HF_REPO_MAP["instruct"]
            model_dir = base_dir / repo_id
            model_dir.mkdir(parents=True)
            (model_dir / "model.vmfb.trim").write_text("model")
            # Manifest matches upstream revision but required files are missing,
            # so this is an "incomplete" (resumable) state, not a stale one.
            write_manifest(model_dir, repo_id, ["model.vmfb.trim"], revision=_REVISION)

            def exists(_repo_id, filename):
                self.assertEqual(_repo_id, repo_id)
                return filename == gemma_setup._GEMMA3_TRIM_LUT_FILENAME

            with (
                mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(gemma_setup, "check_requirements"),
                mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
                mock.patch.object(gemma_setup, "_hf_file_exists", side_effect=exists),
                mock.patch.object(
                    gemma_setup,
                    "download_from_hf",
                    side_effect=_fake_download(base_dir),
                ) as download,
            ):
                gemma_setup.setup_gemma3(["instruct"])

            # The existing model.vmfb.trim is preserved (not re-downloaded).
            downloaded = [call.args[1] for call in download.call_args_list]
            self.assertEqual(
                downloaded,
                [
                    *gemma_setup._GEMMA3_REQUIRED_FILES,
                    gemma_setup._GEMMA3_TRIM_LUT_FILENAME,
                ],
            )
            self.assertTrue((model_dir / "model.vmfb.trim").exists())
            manifest = json.loads((model_dir / ".manifest.json").read_text())
            self.assertEqual(
                manifest["files"],
                [
                    "config.json",
                    "model.vmfb.trim",
                    "token_embeddings.npy",
                    "token_id_lut.npy",
                    "tokenizer.json",
                ],
            )
            self.assertEqual(manifest["revision"], _REVISION)

    def test_gemma_downloads_split_lm_head_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = gemma_setup._HF_REPO_MAP["instruct"]
            model_dir = base_dir / repo_id

            def exists(_repo_id, filename):
                self.assertEqual(_repo_id, repo_id)
                return filename in {"transformer.vmfb", "lm_head.vmfb.trim"}

            with (
                mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(gemma_setup, "check_requirements"),
                mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
                mock.patch.object(gemma_setup, "_hf_file_exists", side_effect=exists),
                mock.patch.object(
                    gemma_setup,
                    "download_from_hf",
                    side_effect=_fake_download(base_dir),
                ) as download,
            ):
                gemma_setup.setup_gemma3(["instruct"])

            downloaded = [call.args[1] for call in download.call_args_list]
            expected_files = [
                "transformer.vmfb",
                "lm_head.vmfb.trim",
                *gemma_setup._GEMMA3_REQUIRED_FILES,
            ]
            self.assertEqual(downloaded, expected_files)
            manifest = json.loads((model_dir / ".manifest.json").read_text())
            self.assertEqual(manifest["files"], sorted(expected_files))
            self.assertEqual(manifest["revision"], _REVISION)

    def test_gemma_repairs_existing_split_body_by_fetching_lm_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = gemma_setup._HF_REPO_MAP["instruct"]
            model_dir = base_dir / repo_id
            model_dir.mkdir(parents=True)
            (model_dir / "transformer.vmfb").write_text("model")
            write_manifest(model_dir, repo_id, ["transformer.vmfb"], revision=_REVISION)

            def exists(_repo_id, filename):
                self.assertEqual(_repo_id, repo_id)
                return filename in {"transformer.vmfb", "lm_head.vmfb"}

            with (
                mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(gemma_setup, "check_requirements"),
                mock.patch.object(gemma_setup, "get_hf_revision", return_value=_REVISION),
                mock.patch.object(gemma_setup, "_hf_file_exists", side_effect=exists),
                mock.patch.object(
                    gemma_setup,
                    "download_from_hf",
                    side_effect=_fake_download(base_dir),
                ) as download,
            ):
                gemma_setup.setup_gemma3(["instruct"])

            downloaded = [call.args[1] for call in download.call_args_list]
            expected_files = [
                "transformer.vmfb",
                "lm_head.vmfb",
                *gemma_setup._GEMMA3_REQUIRED_FILES,
            ]
            self.assertEqual(
                downloaded,
                [
                    "lm_head.vmfb",
                    *gemma_setup._GEMMA3_REQUIRED_FILES,
                ],
            )
            manifest = json.loads((model_dir / ".manifest.json").read_text())
            self.assertEqual(manifest["files"], sorted(expected_files))
            self.assertEqual(manifest["revision"], _REVISION)

    def test_moonshine_skips_when_revision_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = base_dir / repo_id
            files = list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
            for filename in files:
                path = model_dir / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(filename)
            write_manifest(model_dir, repo_id, files, revision=_REVISION)

            with (
                mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(moonshine_setup, "check_requirements"),
                mock.patch.object(moonshine_setup, "get_hf_revision", return_value=_REVISION),
                mock.patch.object(moonshine_setup, "download_from_hf") as download,
            ):
                moonshine_setup.setup_moonshine(["tiny-en"])

            download.assert_not_called()

    def test_moonshine_downloads_required_files_and_records_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = base_dir / repo_id

            with (
                mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(moonshine_setup, "check_requirements"),
                mock.patch.object(moonshine_setup, "get_hf_revision", return_value=_REVISION),
                mock.patch.object(
                    moonshine_setup,
                    "download_from_hf",
                    side_effect=_fake_download(base_dir),
                ) as download,
            ):
                moonshine_setup.setup_moonshine(["tiny-en"])

            downloaded = [call.args[1] for call in download.call_args_list]
            self.assertEqual(
                downloaded,
                list(moonshine_setup._MOONSHINE_REQUIRED_FILES),
            )
            manifest = json.loads((model_dir / ".manifest.json").read_text())
            self.assertEqual(
                manifest["files"],
                sorted(moonshine_setup._MOONSHINE_REQUIRED_FILES),
            )
            self.assertEqual(manifest["revision"], _REVISION)

    def test_moonshine_refreshes_and_clears_stale_files_on_revision_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = base_dir / repo_id
            model_dir.mkdir(parents=True)
            # A complete-but-old copy, including a file dropped from the new set.
            for filename in (*moonshine_setup._MOONSHINE_REQUIRED_FILES, "preprocessor.onnx"):
                (model_dir / filename).write_text("old")
            write_manifest(
                model_dir,
                repo_id,
                [*moonshine_setup._MOONSHINE_REQUIRED_FILES, "preprocessor.onnx"],
                revision="old-revision",
            )

            with (
                mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(moonshine_setup, "check_requirements"),
                mock.patch.object(moonshine_setup, "get_hf_revision", return_value=_REVISION),
                mock.patch.object(
                    moonshine_setup,
                    "download_from_hf",
                    side_effect=_fake_download(base_dir),
                ) as download,
            ):
                moonshine_setup.setup_moonshine(["tiny-en"])

            # All required files re-downloaded after the stale dir was cleared.
            downloaded = [call.args[1] for call in download.call_args_list]
            self.assertEqual(
                downloaded,
                list(moonshine_setup._MOONSHINE_REQUIRED_FILES),
            )
            # The dropped file is gone and content was refreshed.
            self.assertFalse((model_dir / "preprocessor.onnx").exists())
            self.assertEqual((model_dir / "encoder.vmfb").read_text(), "encoder.vmfb")
            manifest = json.loads((model_dir / ".manifest.json").read_text())
            self.assertEqual(manifest["revision"], _REVISION)
            self.assertNotIn("preprocessor.onnx", manifest["files"])

    def _make_moonshine_copy(self, base_dir, repo_id, revision):
        model_dir = base_dir / repo_id
        for filename in moonshine_setup._MOONSHINE_REQUIRED_FILES:
            path = model_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("old")
        write_manifest(model_dir, repo_id, list(moonshine_setup._MOONSHINE_REQUIRED_FILES), revision=revision)
        return model_dir

    def test_inference_refreshes_stale_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = self._make_moonshine_copy(base_dir, repo_id, "old-revision")

            with (
                mock.patch.object(moonshine_setup, "get_hf_revision", return_value=_REVISION),
                mock.patch.object(
                    moonshine_setup,
                    "download_from_hf",
                    side_effect=_fake_download(base_dir),
                ) as download,
            ):
                # model_dir is base/<repo_id>; the helper must recover base_dir.
                moonshine_setup.ensure_moonshine_models(model_dir)

            downloaded = [call.args[1] for call in download.call_args_list]
            self.assertEqual(downloaded, list(moonshine_setup._MOONSHINE_REQUIRED_FILES))
            self.assertEqual((model_dir / "encoder.vmfb").read_text(), "encoder.vmfb")
            manifest = json.loads((model_dir / ".manifest.json").read_text())
            self.assertEqual(manifest["revision"], _REVISION)

    def test_inference_no_refresh_skips_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = self._make_moonshine_copy(base_dir, repo_id, "old-revision")

            with (
                mock.patch.object(moonshine_setup, "get_hf_revision") as revision,
                mock.patch.object(moonshine_setup, "download_from_hf") as download,
            ):
                moonshine_setup.ensure_moonshine_models(model_dir, refresh=False)

            revision.assert_not_called()
            download.assert_not_called()

    def test_inference_without_manifest_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = base_dir / repo_id
            model_dir.mkdir(parents=True)

            with (
                mock.patch.object(moonshine_setup, "get_hf_revision") as revision,
                mock.patch.object(moonshine_setup, "download_from_hf") as download,
            ):
                moonshine_setup.ensure_moonshine_models(model_dir)

            revision.assert_not_called()
            download.assert_not_called()

    def test_moonshine_offline_uses_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = base_dir / repo_id
            files = list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
            for filename in files:
                path = model_dir / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(filename)
            write_manifest(model_dir, repo_id, files, revision="old-revision")

            with (
                mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(moonshine_setup, "check_requirements"),
                mock.patch.object(moonshine_setup, "get_hf_revision", return_value=None),
                mock.patch.object(moonshine_setup, "download_from_hf") as download,
            ):
                moonshine_setup.setup_moonshine(["tiny-en"])

            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
