# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""First-run model resolution and download regressions."""

import argparse
import importlib.util
import logging
from pathlib import Path
from unittest import mock

import pytest

from utils import model_setup
from utils.download import DownloadError, read_manifest, write_manifest
from utils.model_setup import resolve_default_model


def test_existing_model_skips_download(tmp_path):
    download = mock.Mock()
    assert resolve_default_model(
        argparse.ArgumentParser(), lambda: tmp_path, download=download, message="missing"
    ) == tmp_path
    download.assert_not_called()


def test_no_refresh_skips_download(capsys):
    download = mock.Mock()
    with pytest.raises(SystemExit) as exc:
        resolve_default_model(
            argparse.ArgumentParser(), lambda: None,
            download=download, message="missing", no_refresh=True,
        )
    assert exc.value.code == 2
    assert "missing" in capsys.readouterr().err
    download.assert_not_called()


def test_missing_model_downloads_once_then_resolves(tmp_path, caplog):
    resolve = mock.Mock(side_effect=[None, tmp_path])
    download = mock.Mock()
    with caplog.at_level(logging.INFO):
        assert resolve_default_model(
            argparse.ArgumentParser(), resolve, download=download, message="missing"
        ) == tmp_path
    download.assert_called_once_with()
    assert resolve.call_count == 2
    assert "downloading on first run" in caplog.text


def test_download_failure_logs_cause_and_reports_missing_model(caplog, capsys):
    def download():
        raise DownloadError("download failed") from OSError("network down")

    with pytest.raises(SystemExit) as exc:
        resolve_default_model(
            argparse.ArgumentParser(), lambda: None, download=download, message="missing"
        )
    assert exc.value.code == 2
    assert "missing" in capsys.readouterr().err
    assert "download failed" in caplog.text
    assert "network down" in caplog.text


def test_first_run_repair_preserves_manifest_version(tmp_path):
    repo_id = "org/model"
    model_dir = tmp_path / repo_id
    model_dir.mkdir(parents=True)
    model = model_dir / "model.vmfb"
    write_manifest(model_dir, repo_id, [model.name], version="v1", revision="same-sha")

    def download(repo_id, base_dir, *, revision=None):
        (base_dir / repo_id / model.name).write_text(revision)
        return [model.name]

    def fetch():
        return model_setup.download_models(
            {"default": repo_id}, ["default"], base_dir=tmp_path,
            files_present=lambda directory: (directory / model.name).exists(),
            download=download, label="test",
        )

    with mock.patch.object(model_setup, "get_hf_revision", return_value="same-sha") as revision:
        assert resolve_default_model(
            argparse.ArgumentParser(), lambda: model if model.exists() else None,
            download=fetch, message="missing",
        ) == model
    revision.assert_called_once_with(repo_id, revision="v1")
    assert model.read_text() == "v1"
    assert read_manifest(model_dir)["version"] == "v1"


def test_liquid_download_updates_prefill_on_existing_copy(tmp_path):
    path = Path(__file__).resolve().parents[1] / "LiquidAI/LiquidAI-LFM2.5/setup_demo.py"
    spec = importlib.util.spec_from_file_location("liquid_setup", path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    def download(repo_id, filename, *, base_dir, revision=None):
        target = base_dir / repo_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(filename)
        return target

    with (
        mock.patch.object(model_setup, "get_hf_revision", return_value="same-sha"),
        mock.patch.object(setup, "hf_file_exists", return_value=True),
        mock.patch.object(model_setup, "hf_file_exists", new=setup.hf_file_exists),
        mock.patch.object(setup, "download_from_hf", side_effect=download),
        mock.patch.object(model_setup, "download_from_hf", new=setup.download_from_hf),
    ):
        model_dir = setup.download_liquid(base_dir=tmp_path)["default"]
        setup.download_liquid(base_dir=tmp_path, enable_prefill=True)
        prefill = model_dir / "transformer_prefill.vmfb"
        assert prefill.exists()
        assert prefill.name in read_manifest(model_dir)["files"]
        setup.download_liquid(base_dir=tmp_path, enable_prefill=False)
        assert prefill.exists()
        assert prefill.name not in read_manifest(model_dir)["files"]
