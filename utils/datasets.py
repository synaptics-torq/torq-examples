# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validation data from Hugging Face datasets, downloaded on demand.

The repo ships no third-party data: ``src/validate.py`` scripts fetch the
rows they score from the dataset's Hugging Face repo, under the dataset's own
terms (see its dataset card). Downloads are cached under
``<models dir>/datasets/``, so repeat runs are offline.

``fetch_rows`` reads rows through the Hugging Face dataset viewer API, which
returns exactly the rows asked for; media cells (images, audio) are saved as
files by data type and replaced by their local path. ``fetch_file`` downloads
one raw file of a dataset repo, for datasets the viewer does not serve.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import requests

from utils.download import default_models_dir, download_from_hf

ROWS_URL = "https://datasets-server.huggingface.co/rows"
_PAGE = 100  # the viewer API's maximum rows per request
_TIMEOUT = 60
_EXT = {"audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/flac": ".flac", "image/jpeg": ".jpg", "image/png": ".png"}


def datasets_dir() -> Path:
    """Where downloaded validation data is cached."""
    return default_models_dir() / "datasets"


def fetch_file(dataset: str, filename: str, *, revision: str | None = None) -> Path:
    """One raw file of a Hugging Face dataset repo (cached)."""
    return download_from_hf(dataset, filename, base_dir=datasets_dir(), revision=revision, repo_type="dataset")


def _localize(value, stem: Path):
    """Save a media cell (image ``{"src": ...}``, audio ``[{"src": ..., "type": ...}]``)
    as ``stem`` + extension and return the path; other cells come back unchanged."""
    media = value[0] if isinstance(value, list) and value and isinstance(value[0], dict) else value
    if not (isinstance(media, dict) and "src" in media):
        return value
    ext = _EXT.get(media.get("type")) or Path(urlparse(media["src"]).path).suffix or ".bin"
    path = stem.with_name(stem.name + ext)
    response = requests.get(media["src"], timeout=_TIMEOUT)
    response.raise_for_status()
    path.write_bytes(response.content)
    return str(path)


def fetch_rows(dataset: str, *, config: str, split: str, samples: int, offset: int = 0) -> list[dict]:
    """Rows ``offset`` to ``offset + samples - 1`` of a dataset split (fewer at the end of the split).

    Media cells become local file paths. Each row is cached as
    ``<datasets dir>/<dataset>/<config>/<split>/<row>.json`` next to its media.
    """
    cache = datasets_dir() / dataset / config / split
    rows: dict[int, dict] = {}
    missing = []
    for index in range(offset, offset + samples):
        path = cache / f"{index}.json"
        if path.is_file():
            rows[index] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(index)
    if missing:
        cache.mkdir(parents=True, exist_ok=True)
        end = missing[-1] + 1
        for page in range(missing[0], end, _PAGE):
            params = {"dataset": dataset, "config": config, "split": split,
                      "offset": page, "length": min(_PAGE, end - page)}
            response = requests.get(ROWS_URL, params=params, timeout=_TIMEOUT)
            if not response.ok:
                raise RuntimeError(f"Hugging Face dataset viewer: {dataset} ({config}/{split}): "
                                   f"HTTP {response.status_code} {response.text[:200]}")
            entries = response.json().get("rows", [])
            for entry in entries:
                index = entry["row_idx"]
                row = {key: _localize(value, cache / f"{index}_{key}") for key, value in entry["row"].items()}
                (cache / f"{index}.json").write_text(json.dumps(row), encoding="utf-8")
                rows[index] = row
            if len(entries) < params["length"]:
                break  # past the end of the split
    return [rows[index] for index in sorted(rows)]
