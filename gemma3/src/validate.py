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
from collections import Counter
from pathlib import Path

from runner import Gemma3Static
from utils.log import add_logging_args, configure_logging

logger = logging.getLogger("Gemma3.validate")

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "text_translation" / "opus100_en-es.json"


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
        samples: list[dict[str,str]] = data["samples"]
    except KeyError as e:
        raise ValueError(f"Dataset missing required metadata: {e}") from e
    missing = [i for i, d in enumerate(samples) if "src" not in d or "tgt" not in d]
    if missing:
        raise ValueError(
            f"Entries at indices {missing[:5]}{'...' if len(missing) > 5 else ''} "
            "are missing 'src' or 'tgt' keys."
        )
    return samples, src_lang, tgt_lang


def validate(args: argparse.Namespace) -> None:
    configure_logging(args.logging)

    dataset_path = Path(args.dataset)
    logger.info("Loading dataset from %s", dataset_path)
    dataset, src_lang, tgt_lang = load_dataset(dataset_path)

    n = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    logger.info("Evaluating on %d / %d samples", n, len(dataset))

    gemma3 = Gemma3Static(
        args.model,
        args.max_seq_len,
        max_prompt_tokens=args.max_inp_len,
        n_threads=args.threads,
        instruct_model=args.instruct_model,
        cache_keep_n=None if args.no_kv_cache_window else args.kv_cache_window,
        temperature=0.0,   # greedy decoding for reproducibility
        runtime_flags=args.runtime_flags,
    )

    hypotheses: list[str] = []
    references: list[str] = []
    csv_rows: list[dict] = []
    infer_times: list[float] = []
    ttfts: list[float] = []
    toks_per_sec: list[float] = []

    for i, item in enumerate(dataset[:n]):
        src_text: str = item["src"]
        tgt_text: str = item["tgt"]

        sys.stdout.write(f"\r  [{i + 1}/{n}] generating...")
        sys.stdout.flush()

        hypothesis = gemma3.run(f"Translate to {tgt_lang}: \"{src_text}\"").strip().strip('"')

        infer_ms = gemma3.last_infer_time
        ttft_ms = gemma3.time_to_first_token
        decode_ms = infer_ms - ttft_ms
        tps = gemma3.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0.0

        infer_times.append(infer_ms)
        ttfts.append(ttft_ms)
        toks_per_sec.append(tps)

        hypotheses.append(hypothesis)
        references.append(tgt_text)

        if args.verbose:
            print(f"\n  Src: {src_text[:120]}")
            print(f"  Ref: {tgt_text[:120]}")
            print(f"  Hyp: {hypothesis[:120]}  ({infer_ms:.0f} ms, TTFT: {ttft_ms:.0f} ms, {tps:.1f} tok/s)")

        if args.output_csv:
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
        f"\nCorpus BLEU-4: {bleu:.2f}  (n={n})"
        f"\navg {avg_infer:.0f} ms, TTFT: {avg_ttft:.0f} ms, {avg_tps:.1f} tok/s"
    )

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "reference", "hypothesis", "infer_ms", "ttft_ms", "tok_s"])
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Per-sample results written to %s", args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Gemma3 model on a JSON text translation dataset. "
            "Reports corpus BLEU-4."
        )
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to VMFB model"
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
        "--max-seq-len", type=int, default=None,
        help="Maximum sequence length (prompt + generation); auto-detected from model if omitted",
    )
    parser.add_argument(
        "--max-inp-len", type=int,
        help="Maximum prompt token length",
    )
    parser.add_argument(
        "--instruct-model", action="store_true", default=False,
        help="Is instruct model",
    )
    parser.add_argument(
        "-j", "--threads", type=int,
        help="Number of cores to use for CPU execution (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print each source / reference / hypothesis",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None, metavar="FILE",
        help="Write per-sample results to a CSV file",
    )
    inference_group = parser.add_argument_group("inference")
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
    validate(parser.parse_args())
