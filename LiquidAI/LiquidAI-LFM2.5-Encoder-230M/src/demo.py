# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Interactive prompt-routing / fill-mask demo for LFM2.5-Encoder-230M.

Routing mode (default) mirrors the LiquidAI prompt-routing space: type a
prompt, the encoder scores it against the routing lanes in one pass on the
NPU. Lanes are free text and can be edited live:

    /routes                     show lanes
    /add <name> :: <example>    add a lane (example = one in-context anchor)
    /del <name>                 remove a lane
    /mask <text with <|mask|>>  one-off fill-mask query
    Ctrl-D                      exit
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import LiquidEncoderStatic, _DEFAULT_ROUTES  # noqa: E402


def _print_routes(routes: dict[str, str]):
    print("routing lanes:")
    for name, example in routes.items():
        print(f"  - {name}   (e.g. {example!r})" if example else f"  - {name}")


def main(args: argparse.Namespace):
    enc = LiquidEncoderStatic(args.model, seq_len=args.seq_len)
    routes = dict(_DEFAULT_ROUTES)

    if args.prompt:
        results = enc.route(args.prompt, routes)
        for r in results:
            print(f"{r['score']:6.1%}  {r['route']}")
        print(f"({enc.infer_time_ms:.0f} ms encoder pass)")
        return

    _print_routes(routes)
    print("type a prompt to route it (/help for commands)")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/help":
            print(__doc__)
        elif line == "/routes":
            _print_routes(routes)
        elif line.startswith("/add "):
            body = line[5:]
            name, _, example = (s.strip() for s in body.partition("::"))
            if not name:
                print("usage: /add <name> :: <example request>")
                continue
            routes[name] = example or None
            _print_routes(routes)
        elif line.startswith("/del "):
            routes.pop(line[5:].strip(), None)
            _print_routes(routes)
        elif line.startswith("/mask "):
            try:
                for tok, score in enc.fill_mask(line[6:]):
                    print(f"  {score:7.2f}  {tok}")
                print(f"({enc.infer_time_ms:.0f} ms encoder pass)")
            except ValueError as e:
                print(f"error: {e}")
        else:
            results = enc.route(line, routes)
            best = results[0]
            print(f"-> {best['route']}")
            for r in results:
                bar = "#" * int(r["score"] * 40)
                print(f"   {r['score']:6.1%} {r['route']:24s} {bar}")
            print(f"({enc.infer_time_ms:.0f} ms encoder pass)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", required=True,
        metavar=".vmfb | .onnx",
        help="Path to encoder body model (vmfb for Torq, onnx for CPU)")
    parser.add_argument(
        "-p", "--prompt", default=None,
        help="One-shot: route this prompt and exit")
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="Static sequence length of the model (default: from manifest)")
    main(parser.parse_args())
