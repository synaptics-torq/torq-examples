import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gemma3 import setup as gemma_setup
from moonshine import setup as moonshine_setup
from utils.download import write_manifest


def _fake_download(default_base_dir: Path):
    def download(repo_id: str, filename: str, *, base_dir: Path | None = None):
        root = Path(base_dir) if base_dir is not None else default_base_dir
        path = root / repo_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename)
        return path

    return download


class DemoSetupDownloadsTest(unittest.TestCase):
    def test_gemma_skips_valid_manifest(self):
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
            write_manifest(model_dir, repo_id, files)

            with (
                mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(gemma_setup, "check_requirements"),
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

            def exists(_repo_id, filename):
                self.assertEqual(_repo_id, repo_id)
                return filename == gemma_setup._GEMMA3_TRIM_LUT_FILENAME

            with (
                mock.patch.object(gemma_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(gemma_setup, "check_requirements"),
                mock.patch.object(gemma_setup, "_hf_file_exists", side_effect=exists),
                mock.patch.object(
                    gemma_setup,
                    "download_from_hf",
                    side_effect=_fake_download(base_dir),
                ) as download,
            ):
                gemma_setup.setup_gemma3(["instruct"])

            downloaded = [call.args[1] for call in download.call_args_list]
            self.assertEqual(
                downloaded,
                [
                    *gemma_setup._GEMMA3_REQUIRED_FILES,
                    gemma_setup._GEMMA3_TRIM_LUT_FILENAME,
                ],
            )
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

    def test_moonshine_skips_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = base_dir / repo_id
            files = list(moonshine_setup._MOONSHINE_REQUIRED_FILES)
            for filename in files:
                path = model_dir / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(filename)
            write_manifest(model_dir, repo_id, files)

            with (
                mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(moonshine_setup, "check_requirements"),
                mock.patch.object(moonshine_setup, "download_from_hf") as download,
            ):
                moonshine_setup.setup_moonshine(["tiny-en"])

            download.assert_not_called()

    def test_moonshine_prefers_vmfb_preprocessor_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            repo_id = moonshine_setup._HF_REPO_MAP["tiny-en"]
            model_dir = base_dir / repo_id

            def exists(_repo_id, filename):
                self.assertEqual(_repo_id, repo_id)
                return filename in {"preprocessor.vmfb", "preprocessor.onnx"}

            with (
                mock.patch.object(moonshine_setup, "default_models_dir", return_value=base_dir),
                mock.patch.object(moonshine_setup, "check_requirements"),
                mock.patch.object(moonshine_setup, "_hf_file_exists", side_effect=exists),
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
                ["preprocessor.vmfb", *moonshine_setup._MOONSHINE_REQUIRED_FILES],
            )
            manifest = json.loads((model_dir / ".manifest.json").read_text())
            self.assertIn("preprocessor.vmfb", manifest["files"])
            self.assertNotIn("preprocessor.onnx", manifest["files"])


if __name__ == "__main__":
    unittest.main()
