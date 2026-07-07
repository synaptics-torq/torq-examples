# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import sys
from pathlib import Path

from runner import LiquidVLStatic, InferenceInterrupted, DEFAULT_PROMPT
from utils.log import add_logging_args, configure_logging
from utils.terminal import InferenceStopInput

# The model-refresh helper lives one level up (liquidAI-VLM/setup_demo.py). The demo
# dir name has a hyphen, so it is not importable as a package; add it to the path and
# import the module directly. Guarded so a missing setup_demo never breaks inference.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from setup_demo import ensure_lfm2vl_models
except Exception:
    ensure_lfm2vl_models = None

YELLOW = "\033[33m"
RESET = "\033[0m"


def _finish_interrupted_output(started_output: bool) -> None:
    marker = f"{YELLOW}[Interrupt]{RESET}"
    if started_output:
        sys.stdout.write(f" {marker} \n")
    else:
        sys.stdout.write("\r" + " " * 80 + f"\r{marker} \n")
    sys.stdout.flush()


def _print_stats(vl: LiquidVLStatic) -> None:
    # vl.time_to_first_token is only the (text) prefill loop, measured after vision
    # + image-prefill. The real time-to-first-token is all phases summed.
    decode_ms = vl.last_infer_time - vl.time_to_first_token
    tps = vl.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    ttft_ms = vl.vision_time + vl.img_prefill_time + vl.time_to_first_token
    if vl.img_prefill_time > 0:
        # image-prefill path: image tokens seeded in one shot, only text prefilled.
        # img-prefill is forward-only (part inferences + prefix prefill); the vmfb
        # load+free of the image-decoder parts is a one-time cost shown separately.
        phases = (f"vision: {vl.vision_time:.0f} ms, "
                  f"img-prefill: {vl.img_prefill_time:.0f} ms "
                  f"(+{vl.img_load_time:.0f} ms 1-time load), "
                  f"text-prefill: {vl.time_to_first_token:.0f} ms")
    else:
        # standard path: whole prompt (incl. image tokens) prefilled per-token
        phases = (f"vision: {vl.vision_time:.0f} ms, "
                  f"prefill: {vl.time_to_first_token:.0f} ms")
    print(
        f"  ({phases}, TTFT: {ttft_ms:.0f} ms | "
        f"prompt {vl.prefill_tokens} tok, gen: {vl.generated_tokens} tok @ {tps:.1f} tok/s)\n"
    )


def _run_once(vl: LiquidVLStatic, image: str, prompt: str, debug: bool) -> None:
    if debug:
        with InferenceStopInput(sys.stdin) as should_stop:
            answer = vl.run(image, prompt, should_stop=should_stop)
        sys.stdout.write(f"Agent: {answer}")
    else:
        sys.stdout.write('\033[2m[encoding image + thinking...]\033[0m')
        sys.stdout.flush()
        first = True
        with InferenceStopInput(sys.stdin) as should_stop:
            for chunk in vl.run_stream(image, prompt, should_stop=should_stop):
                if first:
                    sys.stdout.write('\r' + ' ' * 40 + '\rAgent: ')
                    first = False
                sys.stdout.write(chunk)
                sys.stdout.flush()
    _print_stats(vl)


def _print_ask_stats(vl: LiquidVLStatic) -> None:
    # Per-question: the image is already cached, so TTFT here is just the question
    # prefill (the vision + image-prefill cost was the one-time begin_image).
    decode_ms = vl.last_infer_time - vl.time_to_first_token
    tps = vl.generated_tokens / decode_ms * 1000 if decode_ms > 0 else 0
    print(
        f"  (TTFT: {vl.time_to_first_token:.0f} ms | prompt {vl.prefill_tokens} tok, "
        f"gen: {vl.generated_tokens} tok @ {tps:.1f} tok/s)\n"
    )


def _ask_once(vl: LiquidVLStatic, question: str, debug: bool) -> None:
    if debug:
        with InferenceStopInput(sys.stdin) as should_stop:
            answer = "".join(vl.ask(question, should_stop=should_stop))
        sys.stdout.write(f"Agent: {answer}")
    else:
        sys.stdout.write('\033[2m[thinking...]\033[0m')
        sys.stdout.flush()
        first = True
        with InferenceStopInput(sys.stdin) as should_stop:
            for chunk in vl.ask(question, should_stop=should_stop):
                if first:
                    sys.stdout.write('\r' + ' ' * 20 + '\rAgent: ')
                    first = False
                sys.stdout.write(chunk)
                sys.stdout.flush()
    _print_ask_stats(vl)


def main(args: argparse.Namespace):
    configure_logging(args.logging)
    if ensure_lfm2vl_models is not None:
        # Verify/refresh the model files against Hugging Face before loading (skipped
        # by --no-refresh; failures are logged, not raised, so offline runs proceed).
        ensure_lfm2vl_models(Path(args.model).parent, refresh=not args.no_refresh)
    logging.getLogger("LiquidVL").info("Loading models...")
    vl = LiquidVLStatic(
        args.model,
        args.vision,
        max_seq_len=args.max_seq_len,
        n_threads=args.threads,
        max_new=args.max_new,
        do_split=not args.no_split,
        native=args.native_res,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        runtime_flags=args.runtime_flags,
        lmhead_path=args.lm_head,
        image_decoder_prefix=args.image_decoder,
        cpu_lm_head=args.cpu_lm_head,
    )
    debug = args.logging.upper() == "DEBUG"

    # The image is a CLI argument (--image). Encode it once, then take questions.
    image = args.image
    if not image:
        try:
            image = input("Image path: ").strip()
        except EOFError:
            return
    if not image or not Path(image).is_file():
        print(f"  {YELLOW}no such image: {image}{RESET}")
        return
    sys.stdout.write('\033[2m[encoding image...]\033[0m')
    sys.stdout.flush()
    try:
        with InferenceStopInput(sys.stdin) as should_stop:
            vl.begin_image(image, should_stop=should_stop)
    except (InferenceInterrupted, KeyboardInterrupt):
        _finish_interrupted_output(False)
        return
    sys.stdout.write('\r' + ' ' * 30 + '\r')
    print(f"  image loaded: {image} "
          f"(vision {vl.vision_time:.0f} ms + img-prefill {vl.img_prefill_time:.0f} ms "
          f"[+{vl.img_load_time:.0f} ms 1-time vmfb load])")

    # one-shot if a question was passed on the CLI (--prompt)
    if args.prompt is not None:
        try:
            _ask_once(vl, args.prompt, debug)
        except (InferenceInterrupted, KeyboardInterrupt):
            _finish_interrupted_output(False)
            _print_ask_stats(vl)
        return

    # interactive: ask questions about --image (reuses the cached image-prefill)
    print("Ask questions about the image ('exit'/'quit' to stop).")
    try:
        while True:
            try:
                q = input("Q: ").strip()
            except EOFError:
                break
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            try:
                _ask_once(vl, q, debug)
            except (InferenceInterrupted, KeyboardInterrupt):
                _finish_interrupted_output(False)
                _print_ask_stats(vl)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LFM2-VL-450M (vision ORT + Torq decoder).")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to the decoder VMFB")
    parser.add_argument("--no-refresh", action="store_true", default=False, dest="no_refresh",
                        help="Skip the Hugging Face model freshness check at startup "
                             "(offline / airgapped runs, e.g. on the board).")
    parser.add_argument("--vision", type=str, required=True,
                        help="Path to the vision_encoder ONNX (run on CPU via onnxruntime)")
    parser.add_argument("--lm-head", type=str, default=None, dest="lm_head",
                        help="Optional standalone lm_head VMFB. When set, -m is the "
                             "decoder *body* (hidden output) and the lm_head is applied "
                             "only when sampling, so prefill tokens skip the "
                             "[1024,65536] lm_head -> lower TTFT.")
    parser.add_argument("--cpu-lm-head", action="store_true", default=False, dest="cpu_lm_head",
                        help="Compute the (tied) lm_head on the host CPU from "
                             "token_embeddings instead of loading lm_head.vmfb. Frees "
                             "one NPU context — recommended with --image-decoder, where "
                             "vision + image parts + decoder already sit at the device "
                             "memory edge.")
    parser.add_argument("--image-decoder", type=str, default=None, dest="image_decoder",
                        help="Image-prefill decoder: either a single full vmfb (path "
                             "ending in .vmfb) or a chain PREFIX that loads the sorted "
                             "{prefix}*.vmfb -- e.g. '.../decoder_image_2part_' "
                             "(-> {A,B}) or '.../decoder_image_3part_' (-> {0,1,2}). "
                             "When set, the 64 image tokens are prefilled in one shot "
                             "instead of 64 per-token decoder calls -> lower TTFT. "
                             "Requires a 256-res image + vision_encoder_256.vmfb.")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path for a one-shot run (omit for the interactive loop)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Ask a single question and exit (one-shot). Omit it to "
                             "enter the interactive multi-question loop on --image.")
    parser.add_argument("--no-split", action="store_true", default=False,
                        help="Disable tiling of large images (single resized image only)")
    parser.add_argument("--native-res", action=argparse.BooleanOptionalAction, default=True,
                        dest="native_res",
                        help="Process the image at its native resolution: no upscale to the "
                             "64-token floor and no patch padding (e.g. 128x128 -> 16 image "
                             "tokens), single-image (no tiling). ON by default (faster). Use "
                             "--no-native-res for the padded 64-token-minimum + tiling "
                             "preprocessing (higher quality on small/large images).")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Decoder KV cache length; auto-detected from the vmfb if omitted")
    parser.add_argument("-j", "--threads", type=int,
                        help="CPU threads for vision ORT + decoder host work (default: all)")
    add_logging_args(parser)
    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-new", type=int, default=64,
                     help="Max tokens to generate per image (default: %(default)s)")
    gen.add_argument("--temperature", type=float, default=0.0,
                     help="Sampling temperature (0.0 = greedy) (default: %(default)s)")
    gen.add_argument("--top-p", type=float, default=1.0,
                     help="Top-p (nucleus) sampling threshold (default: %(default)s)")
    gen.add_argument("--top-k", type=int, default=64,
                     help="Top-k pre-filter size for sampling (default: %(default)s)")
    rt = parser.add_argument_group("runtime")
    rt.add_argument("--runtime-flags", nargs=argparse.REMAINDER, default=None, metavar="FLAG",
                    help="[Advanced] Extra flags for the Torq runtime. Must be specified last.")
    main(parser.parse_args())
