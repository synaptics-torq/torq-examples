# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.metadata
from unittest import mock

import pytest

from utils.deps import MissingRequirementsError, _requirement_name, check_requirements


def test_requirement_name_ignores_comments_options_and_paths():
    assert _requirement_name("") is None
    assert _requirement_name("# comment") is None
    assert _requirement_name("-r other.txt") is None
    assert _requirement_name("./wheelhouse/pkg.whl") is None
    assert _requirement_name("https://example.com/pkg.whl") is None


def test_requirement_name_strips_specifiers_and_extras():
    assert _requirement_name("numpy<2.0") == "numpy"
    assert _requirement_name("tokenizers==0.23.1") == "tokenizers"
    assert _requirement_name("requests[socks]>=2") == "requests"


def test_check_requirements_uses_installed_distributions(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("Pillow\nnumpy<2.0\n")

    with mock.patch(
        "utils.deps.importlib.metadata.distribution",
        return_value=object(),
    ) as distribution:
        check_requirements(req)

    assert [call.args[0] for call in distribution.call_args_list] == ["Pillow", "numpy"]


def test_check_requirements_raises_setup_error_for_missing(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("missing-pkg\n")

    with mock.patch(
        "utils.deps.importlib.metadata.distribution",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        with pytest.raises(MissingRequirementsError):
            check_requirements(req)
