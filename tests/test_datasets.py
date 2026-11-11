# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Tests for ``utils.datasets``: dataset viewer rows are paged, media cells
are saved as files by data type, and cached rows are not fetched again."""

import json
from unittest import mock

import pytest

from utils import datasets


class _Response:
    def __init__(self, payload=None, content=b"", status=200):
        self._payload, self.content, self.status_code = payload, content, status
        self.ok = status < 400
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


def _fake_viewer(total_rows, calls):
    """A viewer with *total_rows* rows: a text cell, an image cell and an audio cell."""
    def get(url, params=None, timeout=None):
        if url != datasets.ROWS_URL:  # a media download
            calls.append(("media", url))
            return _Response(content=url.encode())
        calls.append(("rows", params["offset"], params["length"]))
        end = min(params["offset"] + params["length"], total_rows)
        return _Response({"rows": [
            {"row_idx": i, "row": {
                "text": f"row {i}",
                "image": {"src": f"https://cdn/{i}/image.jpg?Expires=1", "height": 2, "width": 2},
                "audio": [{"src": f"https://cdn/{i}/audio.wav", "type": "audio/wav"}],
            }, "truncated_cells": []}
            for i in range(params["offset"], end)
        ]})
    return get


def test_fetch_rows_pages_saves_media_and_caches(tmp_path):
    calls = []
    with mock.patch.dict("os.environ", {"MODELS": str(tmp_path)}), \
            mock.patch.object(datasets.requests, "get", _fake_viewer(1000, calls)):
        rows = datasets.fetch_rows("org/set", config="c", split="test", samples=150, offset=10)
        assert [c for c in calls if c[0] == "rows"] == [("rows", 10, 100), ("rows", 110, 50)]
        assert len(rows) == 150 and rows[0]["text"] == "row 10"
        image = tmp_path / "datasets/org/set/c/test/10_image.jpg"
        audio = tmp_path / "datasets/org/set/c/test/10_audio.wav"
        assert rows[0]["image"] == str(image) and image.read_bytes() == b"https://cdn/10/image.jpg?Expires=1"
        assert rows[0]["audio"] == str(audio) and audio.is_file()

        calls.clear()
        again = datasets.fetch_rows("org/set", config="c", split="test", samples=150, offset=10)
        assert again == rows and calls == []  # all cached


def test_fetch_rows_stops_at_the_end_of_the_split(tmp_path):
    calls = []
    with mock.patch.dict("os.environ", {"MODELS": str(tmp_path)}), \
            mock.patch.object(datasets.requests, "get", _fake_viewer(120, calls)):
        rows = datasets.fetch_rows("org/set", config="c", split="test", samples=500)
    assert len(rows) == 120
    assert [c for c in calls if c[0] == "rows"] == [("rows", 0, 100), ("rows", 100, 100)]


def test_fetch_rows_reports_viewer_errors(tmp_path):
    with mock.patch.dict("os.environ", {"MODELS": str(tmp_path)}), \
            mock.patch.object(datasets.requests, "get", lambda *a, **k: _Response({"error": "Scan size limit exceeded"}, status=500)):
        with pytest.raises(RuntimeError, match="HTTP 500 .*Scan size limit exceeded"):
            datasets.fetch_rows("org/set", config="c", split="test", samples=1)


def test_fetch_rows_refetches_a_corrupt_cached_row(tmp_path):
    calls = []
    with mock.patch.dict("os.environ", {"MODELS": str(tmp_path)}), \
            mock.patch.object(datasets.requests, "get", _fake_viewer(1000, calls)):
        rows = datasets.fetch_rows("org/set", config="c", split="test", samples=2)
    (tmp_path / "datasets/org/set/c/test/0.json").write_text("")  # truncated cache row
    with mock.patch.dict("os.environ", {"MODELS": str(tmp_path)}), \
            mock.patch.object(datasets.requests, "get", _fake_viewer(1000, calls)):
        calls.clear()
        again = datasets.fetch_rows("org/set", config="c", split="test", samples=2)
    assert again == rows  # the corrupt row came back identical
    assert ("rows", 0, 1) in [c for c in calls if c[0] == "rows"]  # re-fetched only the missing row


def test_fetch_rows_refetches_a_row_whose_media_went_missing(tmp_path):
    calls = []
    with mock.patch.dict("os.environ", {"MODELS": str(tmp_path)}), \
            mock.patch.object(datasets.requests, "get", _fake_viewer(1000, calls)):
        rows = datasets.fetch_rows("org/set", config="c", split="test", samples=2)
    (tmp_path / "datasets/org/set/c/test/1_audio.wav").unlink()
    with mock.patch.dict("os.environ", {"MODELS": str(tmp_path)}), \
            mock.patch.object(datasets.requests, "get", _fake_viewer(1000, calls)):
        calls.clear()
        again = datasets.fetch_rows("org/set", config="c", split="test", samples=2)
    assert again == rows  # the media file was re-downloaded next to the row
    assert ("rows", 1, 1) in [c for c in calls if c[0] == "rows"]  # re-fetched only the affected row
