# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import logging

import pytest

from utils.llm import (
    discover_lm_head_path,
    resolve_lm_head_path,
    resolve_token_id_lut,
)


class TestGemmaLutValidation:
    def test_full_vocab_logits_do_not_need_lut(self):
        assert resolve_token_id_lut(262144, 262144, None) is None

    def test_full_vocab_logits_ignore_unneeded_lut(self, caplog):
        logger = logging.getLogger("test_gemma_lut_validation")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            selected = resolve_token_id_lut(262144, 262144, [1, 2, 3], logger)

        assert selected is None
        assert "ignoring the LUT" in caplog.text

    def test_compact_logits_require_valid_lut(self):
        lut = [10, 20, 30]
        assert resolve_token_id_lut(3, 262144, lut) is lut

    def test_compact_logits_reject_missing_lut(self):
        with pytest.raises(ValueError, match="required"):
            resolve_token_id_lut(3, 262144, None)

    def test_compact_logits_reject_invalid_lut_length(self):
        with pytest.raises(ValueError, match="does not match logits size"):
            resolve_token_id_lut(3, 262144, [10, 20])

    def test_unknown_vocab_still_checks_lut_length_when_possible(self):
        with pytest.raises(ValueError, match="does not match logits size"):
            resolve_token_id_lut(3, None, [10, 20])


class TestGemmaLMHeadDiscovery:
    def test_discovers_single_sibling_lm_head(self, tmp_path):
        model = tmp_path / "model.vmfb"
        lm_head = tmp_path / "model-lm-head.vmfb"
        model.touch()
        lm_head.touch()

        assert discover_lm_head_path(model) == lm_head

    def test_discovers_sibling_lm_head_with_vmfb_suffix(self, tmp_path):
        model = tmp_path / "model.vmfb.trim"
        lm_head = tmp_path / "lm_head.vmfb.w4a16"
        model.touch()
        lm_head.touch()

        assert discover_lm_head_path(model) == lm_head

    def test_discovery_ignores_download_leftovers(self, tmp_path):
        model = tmp_path / "transformer.vmfb"
        lm_head = tmp_path / "lm_head.vmfb.trim"
        model.touch()
        lm_head.touch()
        # A partial download shares the "*.vmfb*" shape but is not loadable.
        (tmp_path / "lm_head.vmfb.incomplete").touch()

        assert discover_lm_head_path(model) == lm_head

    def test_discovery_ignores_model_path_itself(self, tmp_path):
        model = tmp_path / "lm_head_model.vmfb"
        model.touch()

        assert discover_lm_head_path(model) is None

    def test_discovery_rejects_multiple_candidates(self, tmp_path):
        model = tmp_path / "model.vmfb"
        model.touch()
        (tmp_path / "lm_head.vmfb").touch()
        (tmp_path / "other-lm-head.vmfb").touch()

        with pytest.raises(ValueError, match="multiple LM head candidates"):
            discover_lm_head_path(model)

    def test_explicit_lm_head_overrides_discovery(self, tmp_path):
        model = tmp_path / "model.vmfb"
        explicit = tmp_path / "elsewhere.vmfb"
        model.touch()
        explicit.touch()
        (tmp_path / "lm_head.vmfb").touch()

        assert resolve_lm_head_path(model, explicit) == explicit

    def test_disable_lm_head_skips_discovery(self, tmp_path):
        model = tmp_path / "model.vmfb"
        model.touch()
        (tmp_path / "lm_head.vmfb").touch()

        assert resolve_lm_head_path(model, disable_lm_head=True) is None

    def test_rejects_explicit_and_disabled_lm_head(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            resolve_lm_head_path("model.vmfb", "lm_head.vmfb", disable_lm_head=True)
