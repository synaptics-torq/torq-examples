# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.metadata
import re
from pathlib import Path
from unittest import mock

import pytest

from utils.deps import MissingRequirementsError, _requirement_name, check_requirements
from utils.version import examples_version


def test_root_requirements_are_all_parseable():
    # Every line of the root requirements.txt must name a checkable package
    root_req = Path(__file__).resolve().parents[1] / "requirements.txt"
    names = [
        name
        for name in (_requirement_name(line) for line in root_req.read_text().splitlines())
        if name is not None
    ]
    assert "torq-runtime" in names


def test_torq_runtime_pin_matches_examples_version():
    # The torq-runtime pin must track the root VERSION file:
    # a release bump fails here until the pin is updated in the same change.
    root_req = Path(__file__).resolve().parents[1] / "requirements.txt"
    pin = next(
        (line.strip() for line in root_req.read_text().splitlines()
         if _requirement_name(line) == "torq-runtime"),
        None,
    )
    assert pin is not None, "torq-runtime missing from root requirements.txt"
    match = re.match(r"torq-runtime==([^\s#]+)$", pin)
    assert match, f"torq-runtime must be pinned with == in requirements.txt: {pin!r}"
    expected = examples_version().lstrip("v")
    assert match.group(1) == expected, (
        f"torq-runtime pin {match.group(1)!r} does not match VERSION {expected!r}; "
        "update requirements.txt when bumping VERSION."
    )


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
    assert _requirement_name("torq-runtime==2.2.1") == "torq-runtime"


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
