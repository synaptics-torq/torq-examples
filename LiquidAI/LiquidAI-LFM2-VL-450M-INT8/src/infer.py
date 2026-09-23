# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""LFM2-VL-450M W8 (all-NPU) image Q&A.

Encode one image, ask questions about it; switch images mid-session with
``/image <path>``. The model set stays co-resident the whole time (dmabuf
single-copy weights).
"""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm2vl import LFM2VL  # noqa: E402

from utils.log import add_logging_args, configure_logging  # noqa: E402
from utils.terminal import InferenceStopInput  # noqa: E402

YELLOW = "\033[33m"
RESET = "\033[0m"


def _load_bgr(path: str) -> np.ndarray:
    img = Image.open(path)
    rgb = np.asarray(img.convert("RGB"))
    return rgb[:, :, ::-1].copy()  # BGR for the engine


def _print_image_stats(vl: LFM2VL) -> None:
    print(f"  (image: vision {vl.vision_time:.0f} ms, img-prefill {vl.img_prefill_time:.0f} ms, "
          f"prefix {vl.prefix_time:.0f} ms)")


def _print_answer_stats(vl: LFM2VL) -> None:
    s = vl.stats
    print(f"  (TTFT {s['ttft_ms']:.0f} ms | prompt {s['prompt_tokens']} tok, "
          f"gen: {s['n_tokens']} tok @ {s['tok_per_s']:.1f} tok/s)")


def _ask_once(vl: LFM2VL, question: str, debug: bool) -> None:
    if debug:
        with InferenceStopInput(sys.stdin) as should_stop:
            answer = vl.infer(question, should_stop=should_stop)
        print(f"Agent: {answer}")
    else:
        sys.stdout.write('\033[2m[thinking...]\033[0m')
        sys.stdout.flush()
        first = True
        with InferenceStopInput(sys.stdin) as should_stop:
            for chunk in vl.infer_stream(question, should_stop=should_stop):
                if first:
                    sys.stdout.write('\r' + ' ' * 20 + '\rAgent: ')
                    first = False
                sys.stdout.write(chunk)
                sys.stdout.flush()
        print()
    _print_answer_stats(vl)


def main(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    logging.getLogger("LFM2VL").info("Loading models...")
    debug = args.logging.upper() == "DEBUG"

    with LFM2VL(
        args.model_dir,
        decoder=args.model, vision=args.vision, lm_head=args.lm_head,
        image_decoder_prefix=args.image_decoder,
        tda=args.tda, max_new=args.max_new, max_seq_len=args.max_seq_len,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        n_threads=args.threads,
    ) as vl:
        print(f"  loaded: {args.model_dir} (tda={args.tda})")

        def load_image(path: str) -> None:
            if not Path(path).is_file():
                print(f"  {YELLOW}no such image: {path}{RESET}")
                return
            sys.stdout.write('\033[2m[encoding image...]\033[0m')
            sys.stdout.flush()
            with InferenceStopInput(sys.stdin) as should_stop:
                vl.encode_image(_load_bgr(path), should_stop=should_stop)
            sys.stdout.write('\r' + ' ' * 30 + '\r')
            print(f"  image loaded: {path}")
            _print_image_stats(vl)

        image = args.image
        if not image:
            try:
                image = input("Image path: ").strip()
            except EOFError:
                return
            if not image:
                return
        load_image(image)

        # one-shot: a question on the CLI answers and exits
        if args.prompt is not None:
            try:
                _ask_once(vl, args.prompt, debug)
            except RuntimeError as e:
                print(f"  {YELLOW}{e}{RESET}")
            return

        print("Ask questions about the image.")
        print("  /image <path>  encode a new image   /quit  exit")
        while True:
            try:
                line = input("Q: ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ("/quit", "quit", "exit", "q"):
                break
            if line.startswith("/image"):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    print("  usage: /image <path>")
                    continue
                load_image(parts[1].strip())
                continue
            try:
                _ask_once(vl, line, debug)
            except RuntimeError as e:
                print(f"  {YELLOW}{e}{RESET}")
            except KeyboardInterrupt:
                print(f" {YELLOW}[Interrupt]{RESET} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LFM2-VL-450M W8 (all-NPU) image Q&A with the Torq NPU.")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory with the W8 model set (decoder_nolm.vmfb, "
                             "vision_encoder_256.vmfb, lm_head.vmfb, "
                             "decoder_image_2part_*.vmfb, config.json, tokenizer.json, "
                             "token_embeddings.npy)")
    parser.add_argument("--model", type=str, default=None,
                        help="Decoder body vmfb (default: <model-dir>/decoder_nolm.vmfb)")
    parser.add_argument("--vision", type=str, default=None,
                        help="Vision encoder vmfb (default: <model-dir>/vision_encoder_256.vmfb)")
    parser.add_argument("--lm-head", type=str, default=None, dest="lm_head",
                        help="lm_head vmfb (default: <model-dir>/lm_head.vmfb)")
    parser.add_argument("--image-decoder", type=str, default=None, dest="image_decoder",
                        help="Image-prefill chain prefix (default: <model-dir>/decoder_image_2part_)")
    parser.add_argument("--image", type=str, default=None,
                        help="Initial image (omit to be prompted)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Ask a single question and exit (omit for the interactive loop)")
    parser.add_argument("--tda", type=str, choices=["cpu", "dmabuf"], default="dmabuf",
                        help="Allocator backing Torq device buffers (default: %(default)s)")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Decoder KV cache length (auto-detected from the vmfb if omitted)")
    parser.add_argument("-j", "--threads", type=int,
                        help="CPU threads for host work (default: all)")
    add_logging_args(parser)
    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-new", type=int, default=128,
                     help="Max tokens to generate per question (default: %(default)s)")
    gen.add_argument("--temperature", type=float, default=0.0,
                     help="Sampling temperature (0.0 = greedy) (default: %(default)s)")
    gen.add_argument("--top-p", type=float, default=1.0,
                     help="Top-p (nucleus) sampling threshold (default: %(default)s)")
    gen.add_argument("--top-k", type=int, default=64,
                     help="Top-k pre-filter size for sampling (default: %(default)s)")
    main(parser.parse_args())
