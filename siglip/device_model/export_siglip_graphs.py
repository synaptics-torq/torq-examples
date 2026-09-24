#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax[cpu]",
#     "numpy",
#     "safetensors",
# ]
# ///
import sys

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file

from siglip2_jax import (
    HIDDEN,
    NUM_PATCHES,
    PATCH,
    layer_norm,
    linear,
    unflatten_params,
)

DEFAULT_PARAMS = "siglip2_vision_params.safetensors"

def export_function(fn, abstract_args):
    """JIT-lower a function with abstract inputs and write StableHLO text."""
    lowered = jax.jit(fn).lower(*abstract_args)
    hlo_text = lowered.as_text()
    return hlo_text


def export_patchify_matmul(vparams):
    # Input x is [NUM_PATCHES, PATCH * PATCH * 3]
    x_input = jax.ShapeDtypeStruct((NUM_PATCHES, PATCH * PATCH * 3), jnp.bfloat16)
    patch_w = vparams["patch_w"]
    HIDDEN_DIM, C, PH, PW = patch_w.shape
    baked_w = patch_w.transpose(0, 2, 3, 1).reshape(HIDDEN_DIM, PH * PW * C).T
    baked_b = vparams["patch_b"]
    @jax.jit
    def patchify_matmul_baked(x):
        return ((x @ baked_w) + baked_b)
    return export_function(patchify_matmul_baked, (x_input,))


def export_pos_emb_add(pos_emb):
    x_input = jax.ShapeDtypeStruct((NUM_PATCHES, HIDDEN), jnp.bfloat16)
    @jax.jit
    def pos_emb_add_baked(x):
        return x + pos_emb
    return export_function(pos_emb_add_baked, (x_input,))


def export_layer_norm(weight, bias):
    x_input = jax.ShapeDtypeStruct((NUM_PATCHES, HIDDEN), jnp.bfloat16)
    @jax.jit
    def layer_norm_baked(x):
        return layer_norm(x, weight, bias)
    return export_function(layer_norm_baked, (x_input,))


def export_mha_decomposed(block_params):
    x_input = jax.ShapeDtypeStruct((NUM_PATCHES, HIDDEN), jnp.bfloat16)

    q_ws = []
    q_bs = []
    k_ws = []
    k_bs = []
    v_ws = []
    v_bs = []

    for i in range(12):
        start = i * 64
        end = start + 64
        q_ws.append((block_params["q_w"].T)[:, start:end])
        q_bs.append(block_params["q_b"][start:end])

        k_ws.append((block_params["k_w"].T)[:, start:end])
        k_bs.append(block_params["k_b"][start:end])

        v_ws.append((block_params["v_w"].T)[:, start:end])
        v_bs.append(block_params["v_b"][start:end])

    layer_weights = block_params["o_w"].T
    layer_bias = block_params["o_b"]

    @jax.jit
    def mha_decomposed_baked(x):
        x = layer_norm(x, block_params["ln1_w"], block_params["ln1_b"])

        results = []
        for i in range(12):
            q = ((x @ q_ws[i]) + q_bs[i])
            k = ((x @ k_ws[i]) + k_bs[i])
            # Compute attention matrix

            attn = q @ k.T
            scale = 64 ** -0.5
            attn = jax.nn.softmax(scale * attn, axis=-1)

            v = ((x @ v_ws[i]) + v_bs[i])
            results.append(attn @ v)

        result = jnp.concatenate(results, axis=-1)
        return ((result @ layer_weights) + layer_bias)

    return export_function(mha_decomposed_baked, (x_input,))


def export_residual_add():
    x_input = jax.ShapeDtypeStruct((NUM_PATCHES, HIDDEN), jnp.bfloat16)
    r_input = jax.ShapeDtypeStruct((NUM_PATCHES, HIDDEN), jnp.bfloat16)
    @jax.jit
    def residual_add_baked(x, r):
        return x + r
    return export_function(residual_add_baked, (x_input, r_input))


def export_feedforward_residual_a(block_params):
    hidden_inputs = jax.ShapeDtypeStruct((NUM_PATCHES, HIDDEN), jnp.bfloat16)

    @jax.jit
    def feedforward_baked(x):
        x = layer_norm(x, block_params["ln2_w"], block_params["ln2_b"])
        x = linear(x, block_params["fc1_w"], block_params["fc1_b"])
        return x

    return export_function(feedforward_baked, (hidden_inputs,))


def export_gelu():
    x_input = jax.ShapeDtypeStruct((NUM_PATCHES, 3072), jnp.bfloat16)
    @jax.jit
    def gelu_baked(x):
        return jax.nn.gelu(x, approximate=True)
    return export_function(gelu_baked, (x_input,))


def export_feedforward_residual_b(block_params):
    hidden_inputs = jax.ShapeDtypeStruct((NUM_PATCHES, 3072), jnp.bfloat16)
    @jax.jit
    def feedforward_baked_b(x):
        return linear(x, block_params["fc2_w"], block_params["fc2_b"])
    return export_function(feedforward_baked_b, (hidden_inputs,))


def main():
    flat = load_file(DEFAULT_PARAMS)
    vparams = unflatten_params(flat, dtype=jnp.bfloat16)

    patchify_matmul_text = export_patchify_matmul(vparams)
    open("patchify_matmul.stablehlo.mlir", "w").write(patchify_matmul_text)

    pos_emb_add_text = export_pos_emb_add(vparams["pos_emb"])
    open("pos_emb_add.stablehlo.mlir", "w").write(pos_emb_add_text)

    residual_text = export_residual_add()
    open("residual_add.stablehlo.mlir", "w").write(residual_text)

    gelu_text = export_gelu()
    open("gelu.stablehlo.mlir", "w").write(gelu_text)

    for i, block_params in enumerate(vparams["blocks"]):
        mha_residual_text = export_mha_decomposed(block_params)
        open(f"mha_residual_{i}.stablehlo.mlir", "w").write(mha_residual_text)

        feedforward_residual_a_text = export_feedforward_residual_a(block_params)
        open(f"feedforward_residual_a_{i}.stablehlo.mlir", "w").write(
            feedforward_residual_a_text)

        feedforward_residual_b_text = export_feedforward_residual_b(
            block_params)
        open(f"feedforward_residual_b_{i}.stablehlo.mlir", "w").write(
            feedforward_residual_b_text)

    post_layer_norm_text = export_layer_norm(vparams["post_ln_w"],
                                             vparams["post_ln_b"])
    open("post_layer_norm.stablehlo.mlir", "w").write(post_layer_norm_text)


if __name__ == "__main__":
    main()
