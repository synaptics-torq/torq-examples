# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Validate a Gemma3 model against a JSON text translation dataset of source/target pairs.

Dataset format:
    <name>.json
    {
        "src_lang": "English",
        "tgt_lang": "Spanish",
        "samples": [
            {
            "src": "I don't even remember what the fight was about.",
            "tgt": "No recuerdo por qué fue la pelea."
            },
            ...
        ]
    }

Default dataset: opus-100 en-es (data/text_translation/opus100_en-es.json).

Metric: corpus BLEU-4.
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from runner import Gemma3Static
from utils.log import add_logging_args, configure_logging

logger = logging.getLogger("Gemma3.validate")

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "text_translation" / "opus100_en-es.json"

# Default 3-shot examples for English -> Spanish.
DEFAULT_FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    ("Good morning.", "Buenos días."),
    ("Where is the train station?", "¿Dónde está la estación de tren?"),
    ("I would like a glass of water.", "Me gustaría un vaso de agua."),
]

# Regex: preamble the model sometimes prepends before the actual translation.
_PREAMBLE_RE = re.compile(
    r'^\s*'
    r'(?:'
    r"(?:okay[,.]?\s*)?here(?:'s|\s+is|\s+are)\s+(?:the|a|your)?\s*translation[s]?[^:]*[:\-]?"
    r'|traducción[^:]*[:\-]?'
    r'|of\s+["\u201c].+?["\u201d]\s+into\s+\w+\s*[:\-]?'
    r')\s*',
    re.IGNORECASE,
)

# Regex: markdown bold/italic wrappers.
_MD_BOLD_RE = re.compile(r'\*{1,2}(.+?)\*{1,2}')
# Regex: leading bullet / list marker.
_BULLET_RE = re.compile(r'^\s*[\*\-\u2022]\s*')


# ---------------------------------------------------------------------------
# Corpus BLEU-4
# ---------------------------------------------------------------------------

def _tokenize_bleu(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(hypotheses: list[str], references: list[str], max_n: int = 4) -> float:
    """Return corpus BLEU-4 score in [0, 100].

    Implements the original Papineni et al. (2002) formula with a single
    reference per sentence.  Returns 0.0 if any n-gram order has zero
    clipped count.
    """
    if not hypotheses:
        return 0.0

    total_hyp_len = 0
    total_ref_len = 0
    clipped: list[int] = [0] * max_n
    total: list[int] = [0] * max_n

    for hyp, ref in zip(hypotheses, references):
        h = _tokenize_bleu(hyp)
        r = _tokenize_bleu(ref)
        total_hyp_len += len(h)
        total_ref_len += len(r)
        for n in range(1, max_n + 1):
            h_ng = _ngram_counts(h, n)
            r_ng = _ngram_counts(r, n)
            clipped[n - 1] += sum(min(c, r_ng[g]) for g, c in h_ng.items())
            total[n - 1] += max(0, len(h) - n + 1)

    precisions: list[float] = []
    for n in range(max_n):
        if total[n] == 0 or clipped[n] == 0:
            return 0.0
        precisions.append(clipped[n] / total[n])

    log_avg = sum(math.log(p) for p in precisions) / max_n
    if total_hyp_len == 0:
        return 0.0
    bp = 1.0 if total_hyp_len >= total_ref_len else math.exp(1.0 - total_ref_len / total_hyp_len)
    return bp * math.exp(log_avg) * 100.0


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------

def normalize_translation(text: str) -> str:
    """Strip preamble and wrapper artifacts from model output."""
    parts = text.strip().split("\n\n")
    if len(parts) > 1 and _PREAMBLE_RE.search(parts[0]):
        text = "\n\n".join(parts[1:])
    else:
        text = text.strip()
    text = _PREAMBLE_RE.sub("", text).strip()

    first_line = text.split("\n")[0].strip()
    if first_line:
        text = first_line

    text = _BULLET_RE.sub("", text).strip()
    text = _MD_BOLD_RE.sub(r"\1", text).strip()
    for q_open, q_close in [('"', '"'), ('\u201c', '\u201d'), ("'", "'")]:
        if text.startswith(q_open) and text.endswith(q_close) and len(text) > 1:
            text = text[len(q_open):-len(q_close)]
            break

    return text.strip()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str | os.PathLike) -> tuple[list[dict], str, str]:
    """Load a JSON language pair text translation dataset.

    Dataset format:
    <name>.json
    {
        "src_lang": "English",
        "tgt_lang": "Spanish",
        "samples": [
            {
            "src": "I don't even remember what the fight was about.",
            "tgt": "No recuerdo por qué fue la pelea."
            },
            ...
        ]
    }
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Invalid dataset format.")
    try:
        src_lang: str = str(data["src_lang"])
        tgt_lang: str = str(data["tgt_lang"])
        samples: list[dict[str, str]] = data["samples"]
    except KeyError as e:
        raise ValueError(f"Dataset missing required metadata: {e}") from e
    missing = [i for i, d in enumerate(samples) if "src" not in d or "tgt" not in d]
    if missing:
        raise ValueError(
            f"Entries at indices {missing[:5]}{'...' if len(missing) > 5 else ''} "
            "are missing 'src' or 'tgt' keys."
        )
    return samples, src_lang, tgt_lang


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class BaseValidator(ABC):
    """Shared setup for translation validation modes."""

    def __init__(
        self,
        dataset_path: str | os.PathLike,
        gemma3: Gemma3Static,
        *,
        max_samples: int | None = None,
        output_csv: str | os.PathLike | None = None,
        use_few_shot: bool = False,
        few_shot_file: str | os.PathLike | None = None,
        verbose: bool = False,
    ) -> None:
        dataset_path = Path(dataset_path)
        logger.info("Loading dataset from %s", dataset_path)
        self.dataset, self.src_lang, self.tgt_lang = load_dataset(dataset_path)

        self.n = (
            len(self.dataset) if max_samples is None
            else min(max_samples, len(self.dataset))
        )
        logger.info("Evaluating on %d / %d samples", self.n, len(self.dataset))

        self.verbose = verbose
        self.output_csv = output_csv
        self.use_few_shot = use_few_shot
        self.few_shot_file = few_shot_file

        self.gemma3 = gemma3

        # Few-shot prompt setup
        self.few_shot_prompt: str | None = None
        if self.use_few_shot:
            if self.few_shot_file:
                self.few_shot_prompt = Path(self.few_shot_file).read_text()
                logger.info("Loaded few-shot prompt from %s", self.few_shot_file)
            else:
                if self.src_lang.lower() != "english" or self.tgt_lang.lower() != "spanish":
                    raise SystemExit(
                        f"error: --use-few-shot requires --few-shot-file for language pair "
                        f"{self.src_lang}->{self.tgt_lang}. Built-in examples are en->es only."
                    )
                lines = [f"Translate {self.src_lang} to {self.tgt_lang}.", ""]
                for src, tgt in DEFAULT_FEW_SHOT_EXAMPLES:
                    lines.extend([f"{self.src_lang}: {src}", f"{self.tgt_lang}: {tgt}", ""])
                self.few_shot_prompt = "\n".join(lines).rstrip()
                logger.info("Using built-in %d-shot examples (en-es)", len(DEFAULT_FEW_SHOT_EXAMPLES))

    def build_prompt(self, src_text: str) -> str:
        if self.few_shot_prompt is not None:
            if self.few_shot_file:
                return f"{self.few_shot_prompt}{src_text}"
            return f"{self.few_shot_prompt}\n{self.src_lang}: {src_text}\n{self.tgt_lang}:"
        return f"Translate to {self.tgt_lang}: \"{src_text}\""

    def write_csv(self, rows: list[dict], fieldnames: list[str]) -> None:
        if not self.output_csv:
            return
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Per-sample results written to %s", self.output_csv)

    @abstractmethod
    def run(self) -> None: ...


class TextGenerationValidator(BaseValidator):
    """Validates generated translated text with corpus BLEU-4."""

    def run(self) -> None:
        if self.few_shot_prompt:
            logger.info("Few-shot prompting enabled")
        else:
            logger.info("Using direct translation prompt")

        hypotheses: list[str] = []
        references: list[str] = []
        csv_rows: list[dict] = []
        infer_times: list[float] = []
        ttfts: list[float] = []
        toks_per_sec: list[float] = []

        for i, item in enumerate(self.dataset[:self.n]):
            src_text: str = item["src"]
            tgt_text: str = item["tgt"]

            sys.stdout.write(f"\r  [{i + 1}/{self.n}] generating...")
            sys.stdout.flush()

            prompt = self.build_prompt(src_text)
            raw_output = self.gemma3.run(prompt)
            hypothesis = normalize_translation(raw_output)
            hypotheses.append(hypothesis)
            references.append(tgt_text)

            infer_ms = self.gemma3.last_infer_time
            ttft_ms = self.gemma3.time_to_first_token
            decode_ms = infer_ms - ttft_ms
            tps = self.gemma3.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0.0

            infer_times.append(infer_ms)
            ttfts.append(ttft_ms)
            toks_per_sec.append(tps)

            if self.verbose:
                print(
                    f"\n  Src: {src_text[:120]}"
                    f"\n  Ref: {tgt_text[:120]}"
                    f"\n  Hyp: {hypothesis[:120]}"
                    f"  ({infer_ms:.0f} ms, TTFT: {ttft_ms:.0f} ms, {tps:.1f} tok/s)"
                )

            if self.output_csv:
                csv_rows.append({
                    "source": src_text,
                    "reference": tgt_text,
                    "hypothesis": hypothesis,
                    "infer_ms": f"{infer_ms:.0f}",
                    "ttft_ms": f"{ttft_ms:.0f}",
                    "tok_s": f"{tps:.1f}",
                })

        print()

        bleu = corpus_bleu(hypotheses, references)
        avg_infer = sum(infer_times) / len(infer_times) if infer_times else 0.0
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        avg_tps = sum(toks_per_sec) / len(toks_per_sec) if toks_per_sec else 0.0
        print(
            f"\nCorpus BLEU-4: {bleu:.2f}  (n={self.n})"
            f"\navg {avg_infer:.0f} ms, TTFT: {avg_ttft:.0f} ms, {avg_tps:.1f} tok/s"
        )

        self.write_csv(
            csv_rows,
            ["source", "reference", "hypothesis", "infer_ms", "ttft_ms", "tok_s"],
        )


def main(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    gemma3 = Gemma3Static(
        args.model,
        args.max_seq_len,
        max_prompt_tokens=args.max_inp_len,
        n_threads=args.threads,
        instruct_model=args.instruct_model,
        cache_keep_n=None if args.no_kv_cache_window else args.kv_cache_window,
        temperature=0.0,   # greedy decoding for reproducibility
        runtime_flags=args.runtime_flags,
        lm_head_path=args.lm_head,
    )
    val_cls = VALIDATORS[args.mode]
    val_cls(
        args.dataset,
        gemma3,
        max_samples=args.max_samples,
        output_csv=args.output_csv,
        use_few_shot=args.use_few_shot,
        few_shot_file=args.few_shot_file,
        verbose=args.verbose,
    ).run()


if __name__ == "__main__":
    VALIDATORS = {
        "text-generation": TextGenerationValidator,
    }
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Gemma3 model on a JSON text translation dataset. "
            "Supports free-generation (BLEU) and teacher-forced (token accuracy) modes."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="text-generation",
        choices=list(VALIDATORS),
        help="Validation mode (default: %(default)s)",
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to VMFB model"
    )
    parser.add_argument(
        "--lm-head", type=str, default=None, metavar="PATH",
        help="Path to a separately compiled LM head .vmfb (enables split-model inference)",
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET),
        metavar="FILE",
        help=(
            "JSON file: list of {\"src\": ..., \"tgt\": ...} objects. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print per-sample details",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None, metavar="FILE",
        help="Write per-sample results to a CSV file",
    )
    parser.add_argument(
        "--use-few-shot", action="store_true", default=False,
        help=(
            "Use few-shot prompting instead of direct translation prompt. "
            "Requires --few-shot-file for non en->es language pairs."
        ),
    )
    parser.add_argument(
        "--few-shot-file", type=str, default=None, metavar="FILE",
        help=(
            "Text file used as prompt prefix for few-shot prompting. "
            "Contents are used verbatim; the source text is appended. "
            "Only used when --use-few-shot is set. "
            "If omitted, uses built-in 3-shot examples (only for en-es)."
        ),
    )
    inference_group = parser.add_argument_group("inference")
    inference_group.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Maximum sequence length (prompt + generation); auto-detected from model if omitted",
    )
    inference_group.add_argument(
        "--max-inp-len", type=int,
        help="Maximum prompt token length",
    )
    inference_group.add_argument(
        "--instruct-model", action="store_true", default=False,
        help="Is instruct model",
    )
    inference_group.add_argument(
        "-j", "--threads", type=int,
        help="Number of cores to use for CPU execution (default: all)",
    )
    inference_group.add_argument(
        "--kv-cache-window",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Enable sliding-window KV cache: when the cache is full, keep the most "
            "recent N entries and discard older ones before continuing generation "
            "(default: %(default)s)"
        ),
    )
    inference_group.add_argument(
        "--no-kv-cache-window",
        action="store_true",
        default=False,
        help=(
            "Disable sliding-window KV cache behavior. "
            "Once the KV cache reaches its maximum length, no further tokens can be generated."
        ),
    )
    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--runtime-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra flags for the Torq runtime. "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )
    add_logging_args(parser)
    main(parser.parse_args())
