# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Speaker turns from speech regions and windowed speaker embeddings.

All positions are in 16 kHz samples. The pipeline is:

  1. ``speech_regions``   — energy VAD over fixed-size chunks.
  2. ``plan_windows``     — overlapping embedding windows that cover each region.
  3. ``cluster``          — average-linkage clustering of the window embeddings.
  4. ``build_turns``      — per-chunk speaker votes, then runs of one speaker.

Only NumPy is used, so this runs on the board without SciPy or scikit-learn.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Turn:
    start:   int   # first sample
    end:     int   # one past the last sample
    speaker: int   # 0-based speaker index, in order of first appearance


# ── Speech regions ───────────────────────────────────────────────────────────

def speech_regions(audio: np.ndarray, vad, chunk: int, lead_in: int) -> list[tuple[int, int]]:
    """Run ``vad`` over ``audio`` in ``chunk``-sample steps and return the
    voiced regions as (start, end) sample pairs. ``lead_in`` samples of silence
    go first so that a self-calibrating VAD has ambient noise to measure. The
    VAD hangover keeps a region open over short pauses; each region ends at its
    last chunk that is over the threshold, not at the end of the hangover."""
    padded  = np.concatenate([np.zeros(lead_in, dtype=np.float32), audio])
    regions = []
    start = last_voiced = None
    for pos in range(0, len(padded) - chunk + 1, chunk):
        status = vad.process_chunk(padded[pos:pos + chunk])
        at     = pos - lead_in
        if status == "speech_start":
            start = at
        if status in ("speech_start", "speech") and vad.last_score > vad.threshold:
            last_voiced = at + chunk
        if status == "speech_end" and start is not None:
            regions.append((max(start, 0), last_voiced))
            start = None
    if start is not None:
        regions.append((max(start, 0), last_voiced))
    return regions


# ── Embedding windows ────────────────────────────────────────────────────────

def _region_window_starts(start: int, end: int, win: int, hop: int, n: int) -> list[int]:
    """Window starts for one region. A region shorter than a window gets one
    window centred on it; a longer one gets windows every ``hop`` samples, plus
    one that ends at the region end."""
    if end - start <= win:
        centre = (start + end) // 2
        return [int(np.clip(centre - win // 2, 0, max(n - win, 0)))]
    starts = list(range(start, end - win + 1, hop))
    if starts[-1] != end - win:
        starts.append(end - win)
    return starts


def plan_windows(regions: list[tuple[int, int]], win: int, hop: int, n: int) -> np.ndarray:
    """Window start offsets that cover every speech region."""
    starts = [s for start, end in regions for s in _region_window_starts(start, end, win, hop, n)]
    return np.asarray(starts, dtype=np.int64)


# ── Clustering ───────────────────────────────────────────────────────────────

def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def cluster(embeddings: np.ndarray, threshold: float, num_speakers: int | None) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine similarity.

    Merges the two most similar clusters until ``num_speakers`` clusters are
    left, or, when that is ``None``, until no pair of clusters has an average
    similarity above ``threshold``. Returns one label for each row, numbered
    in order of first appearance."""
    sim      = _normalize(embeddings) @ _normalize(embeddings).T
    clusters = [[i] for i in range(len(embeddings))]
    while len(clusters) > (num_speakers or 1):
        best, a, b = _most_similar_pair(sim, clusters)
        if num_speakers is None and best < threshold:
            break
        clusters[a] += clusters.pop(b)
    return _labels_by_first_appearance(clusters, len(embeddings))


def _most_similar_pair(sim: np.ndarray, clusters: list[list[int]]) -> tuple[float, int, int]:
    """The pair of clusters (a < b) with the highest average similarity."""
    best, pair = -np.inf, (0, 1)
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            s = sim[np.ix_(clusters[a], clusters[b])].mean()
            if s > best:
                best, pair = s, (a, b)
    return best, *pair


def _labels_by_first_appearance(clusters: list[list[int]], n: int) -> np.ndarray:
    labels = np.empty(n, dtype=np.int64)
    for label, members in enumerate(sorted(clusters, key=min)):
        labels[members] = label
    return labels


# ── Turns ────────────────────────────────────────────────────────────────────

def build_turns(regions: list[tuple[int, int]], starts: np.ndarray, labels: np.ndarray,
                win: int, chunk: int, min_turn: int) -> list[Turn]:
    """Split each speech region into speaker turns.

    Each window votes for its speaker on every chunk that it covers. Each
    chunk takes the speaker with the most votes, and runs of one speaker
    become turns. A turn shorter than ``min_turn`` samples joins the turn
    before it (or after it, when it is first in its region)."""
    turns = []
    for start, end in regions:
        chunk_starts = np.arange(start, end, chunk)
        speakers     = _vote(chunk_starts, starts, labels, win, chunk)
        turns       += _merge_short(_runs(chunk_starts, speakers, end), min_turn)
    return _renumber(turns)


def _renumber(turns: list[Turn]) -> list[Turn]:
    """Number the speakers 0, 1, ... in order of their first turn. A small
    cluster that wins no chunk vote then leaves no gap in the numbers."""
    order = {}
    for turn in turns:
        turn.speaker = order.setdefault(turn.speaker, len(order))
    return turns


def _vote(chunk_starts: np.ndarray, starts: np.ndarray, labels: np.ndarray,
          win: int, chunk: int) -> np.ndarray:
    """The speaker with the most window votes for each chunk. A window votes on
    a chunk when the chunk centre is inside the window."""
    centres = chunk_starts + chunk // 2
    covers  = (centres[:, None] >= starts[None, :]) & (centres[:, None] < starts[None, :] + win)
    votes   = np.zeros((len(centres), labels.max() + 1))
    for w, label in enumerate(labels):
        votes[covers[:, w], label] += 1
    return votes.argmax(axis=1)


def _runs(chunk_starts: np.ndarray, speakers: np.ndarray, end: int) -> list[Turn]:
    """Consecutive chunks with the same speaker, as turns."""
    turns = []
    for pos, spk in zip(chunk_starts, speakers):
        if turns and turns[-1].speaker == spk:
            continue
        if turns:
            turns[-1].end = int(pos)
        turns.append(Turn(int(pos), end, int(spk)))
    return turns


def _merge_short(turns: list[Turn], min_turn: int) -> list[Turn]:
    """Join each turn that is shorter than ``min_turn`` to a neighbour."""
    merged = []
    for turn in turns:
        if merged and (turn.end - turn.start < min_turn or merged[-1].speaker == turn.speaker):
            merged[-1].end = turn.end
        else:
            merged.append(turn)
    if len(merged) > 1 and merged[0].end - merged[0].start < min_turn:
        merged[1].start = merged[0].start
        merged.pop(0)
    return merged
