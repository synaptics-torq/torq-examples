#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax[cpu]",
#     "torch",
#     "transformers>=4.49",
#     "safetensors",
# ]
# ///
"""
Export SigLIP2 vision encoder parameters to a safetensors file.

Loads the model via transformers, extracts vision encoder weights into
JAX arrays (bfloat16), flattens the nested param dict, and writes them
to a safetensors file.
"""

import argparse
import sys

import jax.numpy as jnp
import torch
from safetensors.flax import save_file
from transformers import AutoModel

from siglip2_jax import extract_vision_params

MODEL_ID = "google/siglip2-base-patch32-256"
DEFAULT_OUTPUT = "siglip2_vision_params.safetensors"


def flatten_params(params, prefix=""):
    """Flatten nested param dict into {dotted.key: jax array} pairs."""
    flat = {}
    for k, v in params.items():
        full_key = f"{prefix}{k}" if prefix else k
        if k == "blocks":
            for i, blk in enumerate(v):
                flat.update(flatten_params(blk, prefix=f"blocks.{i}."))
        elif isinstance(v, dict):
            flat.update(flatten_params(v, prefix=f"{full_key}."))
        else:
            flat[full_key] = v
    return flat


def main():
    parser = argparse.ArgumentParser(description="Export SigLIP2 vision parameters.")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="HuggingFace model ID.")
    parser.add_argument("--weights", type=str, help="Path to local pytorch_model.bin or similar.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output safetensors path.")
    args = parser.parse_args()

    if args.weights:
        print(f"Loading local weights from {args.weights} …")
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    else:
        print(f"Loading {args.model_id} via transformers …")
        model = AutoModel.from_pretrained(args.model_id, dtype=torch.float32)
        model.eval()
        state = model.state_dict()

    print("Extracting vision encoder parameters (bf16) …")
    vparams = extract_vision_params(state, dtype=jnp.bfloat16)

    print("Flattening parameters …")
    flat = flatten_params(vparams)

    print(f"Writing {len(flat)} tensors to {args.output} …")
    save_file(flat, args.output)

    total = sum(a.size for a in flat.values())
    print(f"Done — {total:,} parameters saved.")


if __name__ == "__main__":
    main()
