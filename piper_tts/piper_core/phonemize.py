# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Text -> Piper phoneme ids, via a persistent espeak-ng daemon.

``phonemizerd`` (a small C daemon, source in ``phonemizerd.c``) mirrors
libpiper's ``piper_synthesize_start()`` — same espeak call, same clause
terminator classification, same punctuation append. This module mirrors the id
mapping that follows it: NFD-normalize, drop ``(lang)`` switches, and emit
``[BOS, PAD, (id, PAD)*, EOS]`` per sentence.

espeak is loaded once and stays resident, so per-utterance phonemization is a
few milliseconds instead of a fresh dictionary load.
"""

import json
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

ID_BOS, ID_EOS, ID_PAD = 1, 2, 0
VOICE_JSON = "voice/en_US-libritts_r-medium.onnx.json"


class Phonemizer:
    """Persistent espeak-ng phonemizer producing Piper-exact id sequences."""

    def __init__(self, model_dir):
        d = Path(model_dir)
        daemon, data = d / "espeak" / "phonemizerd", d / "espeak" / "espeak-ng-data"
        for p in (daemon, data, d / VOICE_JSON):
            if not p.exists():
                raise FileNotFoundError(f"missing phonemizer asset: {p}")
        self.id_map = json.loads((d / VOICE_JSON).read_text())["phoneme_id_map"]
        self.proc = subprocess.Popen([str(daemon), str(data), "en-us"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        if self.proc.stdout.readline().strip() != "READY":
            raise RuntimeError(f"{daemon} did not start")
        self("Warm up.")  # first call primes espeak's dictionaries

    def _sentences(self, text):
        """Ask the daemon for one IPA string per sentence."""
        self.proc.stdin.write(text.replace("\n", " ") + "\n")
        self.proc.stdin.flush()
        out, cur = [], ""
        while True:
            line = self.proc.stdout.readline()
            if not line or line.strip() == "DONE":
                break
            _, sentence_end, phonemes = line.rstrip("\n").split("\t", 2)
            cur += phonemes
            if sentence_end == "1":
                out.append(cur) if cur.strip() else None
                cur = ""
        if cur.strip():
            out.append(cur)
        return out

    def _ids(self, sentence):
        """Map one IPA sentence to Piper's id sequence (PAD-interleaved)."""
        ids, in_lang = [ID_BOS, ID_PAD], False
        for cp in unicodedata.normalize("NFD", sentence):
            if in_lang:                      # inside a "(en)" language switch
                in_lang = cp != ")"
            elif cp == "(":
                in_lang = True
            else:
                for i in self.id_map.get(cp, []):
                    ids.extend((i, ID_PAD))
        return np.array([*ids, ID_EOS], dtype=np.int64)

    def __call__(self, text):
        """Return a list of int64 id arrays, one per sentence."""
        return [self._ids(s) for s in self._sentences(text)]

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close(), self.proc.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
