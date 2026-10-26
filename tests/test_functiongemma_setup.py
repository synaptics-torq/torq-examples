# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from unittest import mock

from functiongemma import setup_demo as setup
from gemma3.setup_demo import GEMMA3_HF_REPO_MAP
import setup_demos


def test_functiongemma_default_is_separate_from_gemma(tmp_path):
    repo = setup.FUNCTIONGEMMA_HF_REPO_MAP[setup.DEFAULT_MODEL]
    assert repo not in GEMMA3_HF_REPO_MAP.values()
    assert setup.local_functiongemma_model_path(base_dir=tmp_path) is None
    model_dir = tmp_path / repo
    model_dir.mkdir(parents=True)
    (model_dir / "transformer.vmfb").touch()
    assert setup.local_functiongemma_model_path(base_dir=tmp_path) == model_dir / "transformer.vmfb"


def test_root_setup_forwards_functiongemma_options():
    assert "functiongemma" in setup_demos.DEMOS
    with mock.patch.object(setup, "setup_functiongemma") as run:
        setup_demos.setup_demo("functiongemma", model_version="v2.3.0", no_update=True, enable_prefill=True)
    run.assert_called_once_with(model_version="v2.3.0", no_update=True, enable_prefill=True)


def test_functiongemma_setup_uses_own_requirements_and_shared_layout(tmp_path):
    with mock.patch.object(setup, "setup_demo", return_value={}) as run:
        setup.setup_functiongemma(model_version="v2.3.0", no_update=True, enable_prefill=True)
    args, kwargs = run.call_args
    assert args == (setup.FUNCTIONGEMMA_HF_REPO_MAP, [setup.DEFAULT_MODEL])
    assert kwargs["demo_name"] == "functiongemma"
    assert kwargs["requirements"].parent.name == "functiongemma"
    assert kwargs["model_version"] == "v2.3.0"
    assert kwargs["no_update"] is True
    assert not kwargs["files_present"](tmp_path)
    assert kwargs["download"] is setup._download_gemma3
    assert kwargs["enable_prefill"] is True
