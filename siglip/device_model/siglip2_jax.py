"""
Pure JAX implementation of the SigLIP2 vision encoder (base-patch32-256).

Provides a JIT-compiled forward pass that produces 64 image tokens (8x8 grid)
of dimension 768. Weights are extracted from a HuggingFace transformers state dict.
"""

import jax
import jax.numpy as jnp

# --- Architecture constants ------------------------------------------
HIDDEN = 768
HEADS = 12
HEAD_DIM = HIDDEN // HEADS  # 64
LAYERS = 12
PATCH = 32
IMAGE_SIZE = 256
NUM_PATCHES = (IMAGE_SIZE // PATCH) ** 2  # 64
LN_EPS = 1e-6


# --- Pure JAX helpers ------------------------------------------------

def layer_norm(x, weight, bias):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    diff = x - mean
    var = jnp.mean(diff * diff, axis=-1, keepdims=True)
    return diff / jnp.sqrt(var + LN_EPS) * weight + bias


def gelu_tanh(x):
    return jax.nn.gelu(x, approximate=True)


def gelu_pade(x):
    """
    Computes GELU(x) using an optimized 4th-order Pade approximation of the 
    ReLU-GELU residual. 
    
    Identity: GELU(x) = ReLU(x) - |x|*Phi(-|x|)
    """
    abs_x = jnp.abs(x)

    # Optimized 3rd-Order Pade Coefficients
    a = -0.276821
    b, c, d = 2.436267, -3.064979, 3.097303

    # Rational approximation of the residual
    # num = |x| * (1 + a|x|)
    # den = 2 + b|x| + c|x|^2 + d|x|^3
    num = abs_x * (1 + a * abs_x)
    den = 2.0 + b * abs_x + c * abs_x**2 + d * abs_x**3
    residual = num / den

    # GELU(x) = max(0, x) - residual
    return jax.nn.relu(x) - residual



def linear(x, weight, bias):
    """weight shape: [out, in] (PyTorch convention) - we transpose."""
    return x @ weight.T + bias


def single_head_attention(x, q_w, q_b, k_w, k_b, v_w, v_b):
    """Computes attention for a single head."""
    # Project inputs to queries, keys, and values for this specific head
    q = linear(x, q_w, q_b)
    k = linear(x, k_w, k_b)
    v = linear(x, v_w, v_b)

    scale = HEAD_DIM ** -0.5
    
    # Compute scaled dot-product attention
    attn = (q @ k.T) * scale
    attn = jax.nn.softmax(attn, axis=-1)
    return attn @ v


def mha_decomposed(x, q_w, q_b, k_w, k_b, v_w, v_b):
    """Multi-head attention with separate Q/K/V projections."""
    head_outs = []
    for i in range(HEADS):
        start_idx = i * HEAD_DIM
        end_idx = (i + 1) * HEAD_DIM
        
        # Slice weights and biases for the i-th head
        q_w_i = q_w[start_idx:end_idx]
        q_b_i = q_b[start_idx:end_idx]
        k_w_i = k_w[start_idx:end_idx]
        k_b_i = k_b[start_idx:end_idx]
        v_w_i = v_w[start_idx:end_idx]
        v_b_i = v_b[start_idx:end_idx]
        head_out = single_head_attention(
            x, q_w_i, q_b_i, k_w_i, k_b_i, v_w_i, v_b_i)
        head_outs.append(head_out)

    return jnp.concatenate(head_outs, axis=-1)


def mha(x, q_w, q_b, k_w, k_b, v_w, v_b):
    """Multi-head attention with separate Q/K/V projections."""
    L_q, _ = x.shape
    L_k, _ = x.shape

    q = linear(x, q_w, q_b).reshape(L_q, HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = linear(x, k_w, k_b).reshape(L_k, HEADS, HEAD_DIM).transpose(1, 2, 0)
    v = linear(x, v_w, v_b).reshape(L_k, HEADS, HEAD_DIM).transpose(1, 0, 2)

    scale = HEAD_DIM ** -0.5
    attn = (q @ k) * scale
    attn = jax.nn.softmax(attn, axis=-1)
    out = (attn @ v).transpose(1, 0, 2).reshape(L_q, HIDDEN)
    return out


def feedforward_residual(x, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b):
    x = layer_norm(x, ln_w, ln_b)
    x = linear(x, fc1_w, fc1_b)

    # TODO(derekjchow): Use GELU here once torq supports it
    x = jax.nn.gelu(x, approximate=True)
    # x = jax.nn.relu(x)
    # x = gelu_pade(x)

    x = linear(x, fc2_w, fc2_b)
    return x


def feedforward(x, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b):
    """Feedforward component of the transformer block."""
    residual = x
    x = feedforward_residual(x, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b)
    # x = layer_norm(x, ln_w, ln_b)
    # x = linear(x, fc1_w, fc1_b)

    # # TODO(derekjchow): Use GELU here once torq supports it
    # x = jax.nn.gelu(x, approximate=True)
    # # x = jax.nn.relu(x)
    # # x = gelu_pade(x)

    # x = linear(x, fc2_w, fc2_b)
    return residual + x


def mha_subblock(x, ln_w, ln_b, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b):
    """Multihead attention subblock with layer-norm, residual and projection."""
    residual = x
    x = layer_norm(x, ln_w, ln_b)
    x = mha_decomposed(x, q_w, q_b, k_w, k_b, v_w, v_b)
    x = linear(x, o_w, o_b)
    return residual + x


def transformer_block(x, params):
    """Pre-norm transformer encoder block."""
    x = mha_subblock(x, params["ln1_w"], params["ln1_b"],
                    params["q_w"], params["q_b"],
                    params["k_w"], params["k_b"],
                    params["v_w"], params["v_b"],
                    params["o_w"], params["o_b"]
                    )
    # residual = x
    # x = layer_norm(x, params["ln1_w"], params["ln1_b"])
    # # x = mha(x,
    # x = mha_decomposed(x,
    #         params["q_w"], params["q_b"],
    #         params["k_w"], params["k_b"],
    #         params["v_w"], params["v_b"])
    # x = linear(x, params["o_w"], params["o_b"])
    # x = residual + x

    x = feedforward(x, params["ln2_w"], params["ln2_b"],
                    params["fc1_w"], params["fc1_b"],
                    params["fc2_w"], params["fc2_b"])
    return x


def transformer_tower(x, all_block_params):
    for block_params in all_block_params:
        x = transformer_block(x, block_params)
    return x


# --- Exportable subgraphs -------------------------------------------

def space2depth_reshape(pixel_values):
    """Space2depth transformation and reshape: [C, H, W] -> [N, P*P*C]."""
    C, H, W = pixel_values.shape
    x = pixel_values.transpose(1, 2, 0)  # [H, W, C]
    x = x.reshape(H // PATCH, PATCH, W // PATCH, PATCH, C)
    x = x.transpose(0, 2, 1, 3, 4)  # [H/P, W/P, PATCH, PATCH, C]
    x = x.reshape(NUM_PATCHES, PATCH * PATCH * C)
    return x


def patchify_matmul(x, patch_w, patch_b, pos_emb=None):
    """Linear projection for patch embedding: [N, P*P*C] -> [N, HIDDEN]."""
    # patch_w shape can be [HIDDEN, C, PATCH, PATCH] or [HIDDEN, PATCH*PATCH*C]
    if patch_w.ndim == 4:
        HIDDEN_DIM, C, PH, PW = patch_w.shape
        w = patch_w.transpose(0, 2, 3, 1).reshape(HIDDEN_DIM, PH * PW * C)
    else:
        w = patch_w
    
    x = (x @ w.T) + patch_b
    # if pos_emb is not None:
    #     x = x + pos_emb
    x = x + pos_emb
    return x


def patchify_stem(pixel_values, patch_w, patch_b, pos_emb):
    """Patch embedding (space2depth + matmul) + position embeddings."""
    # Apply to batch using vmap
    x = jax.vmap(space2depth_reshape)(pixel_values)
    x = jax.vmap(lambda x: patchify_matmul(x, patch_w, patch_b, pos_emb))(x)
    return x


def patchify_stem_nhwc(pixel_values, kernel, bias, pos_emb=None):
    H, W, C = pixel_values.shape
    # Space2depth transformation
    x = pixel_values.reshape(H // PATCH, PATCH, W // PATCH, PATCH, C)
    x = x.transpose(0, 2, 1, 3, 4)  # [H/P, W/P, PATCH, PATCH, C]
    x = x.reshape(NUM_PATCHES, PATCH * PATCH * C)

    # Matrix multiplication (linear projection)
    # kernel is already OHWI: [HIDDEN, PATCH, PATCH, C]
    w = kernel.reshape(HIDDEN, PATCH * PATCH * C)
    x = x @ w.T + bias

    if pos_emb is not None:
        x = x + pos_emb
    return x


def patchify_stem_pres2d(x, kernel, bias, pos_emb):
    # Assuming x has already been transformed by space2depth and reshaped.
    return linear(x, kernel, bias) + pos_emb


def attention_half(x, ln1_w, ln1_b, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b):
    """LN + MHA + residual: [B,N,768] -> [B,N,768]."""
    residual = x
    x = layer_norm(x, ln1_w, ln1_b)
    x = mha(x, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b)
    return residual + x


def attn_qkv_proj(x, ln1_w, ln1_b, q_w, q_b, k_w, k_b, v_w, v_b):
    """LN + Q/K/V projections: [B,N,768] -> 3x [B,12,N,64]."""
    B, L, _ = x.shape
    x = layer_norm(x, ln1_w, ln1_b)
    q = linear(x, q_w, q_b).reshape(B, L, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = linear(x, k_w, k_b).reshape(B, L, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = linear(x, v_w, v_b).reshape(B, L, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    return q, k, v


def attn_scores_and_combine(q, k, v):
    """Attention scores + weighted sum: [B,12,N,64] -> [B,N,768]."""
    scale = HEAD_DIM ** -0.5
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    attn = jax.nn.softmax(attn, axis=-1)
    B = q.shape[0]
    L = q.shape[2]
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, HIDDEN)
    return out


def attn_output_proj(attn_out, residual, o_w, o_b):
    """Output projection + residual: [B,N,768] -> [B,N,768]."""
    return residual + linear(attn_out, o_w, o_b)


def mlp_half(x, ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b):
    """LN + MLP + residual: [B,196,768] -> [B,196,768]."""
    residual = x
    x = layer_norm(x, ln2_w, ln2_b)
    x = linear(x, fc1_w, fc1_b)
    x = gelu_tanh(x)
    x = linear(x, fc2_w, fc2_b)
    return residual + x


def post_layernorm(x, weight, bias):
    """Final layer normalization: [B,196,768] -> [B,196,768]."""
    return layer_norm(x, weight, bias)


# --- Vision encoder (tokens only, no pooling) -----------------------

@jax.jit
def vision_tokens(pixel_values, vparams):
    """Vision tower up to post-layernorm: returns [B, 64, D] tokens."""
    def forward_single(pixels):
        # Patch embedding (space2depth + matmul)
        x = space2depth_reshape(pixels)
        x = patchify_matmul(x, vparams["patch_w"], vparams["patch_b"], vparams["pos_emb"])

        # Transformer encoder
        for blk in vparams["blocks"]:
            x = transformer_block(x, blk)

        # Post-layernorm
        x = layer_norm(x, vparams["post_ln_w"], vparams["post_ln_b"])

        # Optional projection layer
        if "proj_w" in vparams:
            x = linear(x, vparams["proj_w"], vparams["proj_b"])

        return x

    return jax.vmap(forward_single)(pixel_values)


# --- Param (de)serialization -----------------------------------------

def unflatten_params(flat: dict, dtype=jnp.float32) -> dict:
    """Reconstruct the nested vparams dict from flat dotted keys."""
    blocks = [{} for _ in range(LAYERS)]
    top = {}
    for key, arr in flat.items():
        val = jnp.array(arr, dtype=dtype)
        if key.startswith("blocks."):
            parts = key.split(".", 2)  # ["blocks", "3", "ln1_w"]
            blocks[int(parts[1])][parts[2]] = val
        else:
            top[key] = val
    top["blocks"] = blocks
    return top


# --- Weight extraction (requires torch) --------------------------------

def _to_jax(t, dtype=jnp.float32) -> jnp.ndarray:
    import torch  # noqa: F811 — lazy import, only needed for weight extraction
    return jnp.array(t.detach().float().numpy(), dtype=dtype)


def _extract_block_params(state, prefix, dtype=jnp.float32):
    j = lambda t: _to_jax(t, dtype)
    return {
        "ln1_w": j(state[f"{prefix}.layer_norm1.weight"]),
        "ln1_b": j(state[f"{prefix}.layer_norm1.bias"]),
        "q_w": j(state[f"{prefix}.self_attn.q_proj.weight"]),
        "q_b": j(state[f"{prefix}.self_attn.q_proj.bias"]),
        "k_w": j(state[f"{prefix}.self_attn.k_proj.weight"]),
        "k_b": j(state[f"{prefix}.self_attn.k_proj.bias"]),
        "v_w": j(state[f"{prefix}.self_attn.v_proj.weight"]),
        "v_b": j(state[f"{prefix}.self_attn.v_proj.bias"]),
        "o_w": j(state[f"{prefix}.self_attn.out_proj.weight"]),
        "o_b": j(state[f"{prefix}.self_attn.out_proj.bias"]),
        "ln2_w": j(state[f"{prefix}.layer_norm2.weight"]),
        "ln2_b": j(state[f"{prefix}.layer_norm2.bias"]),
        "fc1_w": j(state[f"{prefix}.mlp.fc1.weight"]),
        "fc1_b": j(state[f"{prefix}.mlp.fc1.bias"]),
        "fc2_w": j(state[f"{prefix}.mlp.fc2.weight"]),
        "fc2_b": j(state[f"{prefix}.mlp.fc2.bias"]),
    }


def extract_vision_params(state, dtype=jnp.float32):
    """Extract vision encoder weights from a state dict into JAX arrays.

    Supports both standard Siglip2VisionModel and Gemma3Nano-style state dicts.
    """
    j = lambda t: _to_jax(t, dtype)

    # Identify prefix (e.g. "vision_model." or "vision_tower.vision_model.")
    prefix = ""
    if "vision_model.embeddings.patch_embedding.weight" in state:
        prefix = "vision_model."
    elif "vision_tower.vision_model.embeddings.patch_embedding.weight" in state:
        prefix = "vision_tower.vision_model."
    else:
        # Try to find any key that looks like it belongs to the vision tower
        for key in state.keys():
            if "embeddings.patch_embedding.weight" in key:
                prefix = key.replace("embeddings.patch_embedding.weight", "")
                break

    blocks = [
        _extract_block_params(state, f"{prefix}encoder.layers.{i}", dtype)
        for i in range(LAYERS)
    ]

    params = {
        "patch_w": j(state[f"{prefix}embeddings.patch_embedding.weight"]),
        "patch_b": j(state[f"{prefix}embeddings.patch_embedding.bias"]),
        "pos_emb": j(state[f"{prefix}embeddings.position_embedding.weight"]),
        "blocks": blocks,
        "post_ln_w": j(state[f"{prefix}post_layernorm.weight"]),
        "post_ln_b": j(state[f"{prefix}post_layernorm.bias"]),
    }

    # Extract projector if present (usually "projector" or "vision_tower.projector")
    proj_prefix = prefix.replace("vision_model.", "") if "vision_model." in prefix else ""
    if f"{proj_prefix}projector.weight" in state:
        params["proj_w"] = j(state[f"{proj_prefix}projector.weight"])
        params["proj_b"] = j(state[f"{proj_prefix}projector.bias"])

    return params
