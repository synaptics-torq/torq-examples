# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Streaming Moonshine engine (2-split VMFB) — committed-prefix incremental decode.

Orchestrates two VMFB sessions (a fused ``encoder`` and a ``decoder_kv``) for
static streaming inference. Model artifacts are read from a single flat model
directory (``models/Synaptics/moonshine-streaming-tiny-torq/``):

  * ``encoder.vmfb`` / ``decoder.vmfb``      — the quantized Torq builds
  * ``streaming_config.json`` / ``config.json``
  * ``adapter_pos_emb.npy`` / ``decoder_token_embeddings.npy`` / ``tokenizer.json``

The decoder resumes from a committed prefix instead of re-decoding from BOS every
preview (O(T) instead of O(T^2)). A token is committed (its self-KV frozen and
reused) only when BOTH:

  1. LocalAgreement-N: it is identical across the last N hypotheses, and
  2. it is at least ``commit_delay_sec`` of audio behind the live frontier.

The decoder cross-attention is global, so a committed token's self-KV is computed
against a smaller memory and is mildly stale; the two gates above only freeze
tokens that have stopped changing AND are well behind the frontier, so the drift
is negligible. ``decode()`` restores the baseline re-decode-from-BOS behaviour.
"""

import json
import logging
import math
import os

import numpy as np

from torq.runtime import VMFBInferenceRunner
from iree.runtime import DeviceArray

logger = logging.getLogger("moonshine_streaming")

# This tree's model basenames (== the demo's fused_encoder / decoder_kv).
ENCODER_NAME = "encoder"
DECODER_NAME = "decoder"

# A VMFB exposes its inputs positionally (no argument names), so the dict-based
# feed interface needs to know each model's input order. These lists are the
# canonical order baked into the compiled VMFBs, pinned to the
# moonshine-streaming-tiny export (6 decoder layers, 6 encoder buffers). A
# re-export with a different arity trips the arity check in ``_named_inputs_info``;
# a pure reordering would not, so keep these in sync with the model if it is rebuilt.
ENCODER_INPUT_ORDER = [
    "audio_chunk", "conv1_buffer", "conv2_buffer", "features_buffer",
    "position_embeddings",
    *(f"buf_{i}" for i in range(6)),
]
DECODER_INPUT_ORDER = [
    "inputs_embeds",
    *(nm for i in range(6) for nm in (f"k_self_{i}", f"v_self_{i}")),
    *(nm for i in range(6) for nm in (f"k_cross_{i}", f"v_cross_{i}")),
    "cross_attn_bias", "position_ids", "current_len",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _common_prefix_len(a: list, b: list) -> int:
    """Length of the longest shared prefix of two token lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _agree_prefix_len(hyps: list) -> int:
    """Longest prefix length shared by ALL token lists (LocalAgreement-N)."""
    if not hyps:
        return 0
    cp = len(hyps[0])
    for h in hyps[1:]:
        cp = min(cp, _common_prefix_len(hyps[0], h))
    return cp


def find_asset(model_dir: str, filename: str) -> str:
    """Return the path to ``filename`` inside the flat model directory."""
    path = os.path.join(model_dir, filename)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"Cannot find '{filename}' in {model_dir}."
    )


def _load_bfloat16_table(path: str) -> np.ndarray:
    """Load a bf16 NumPy table stored using NumPy's two-byte void dtype."""
    arr = np.load(path)
    if arr.dtype == np.dtype("V2"):
        import ml_dtypes

        arr = arr.view(ml_dtypes.bfloat16)
    return arr


# ── VMFB input/output helpers ─────────────────────────────────────────────────
#
# A VMFBInferenceRunner exposes its model's inputs positionally (no argument
# names); these helpers pair that positional interface with the hardcoded
# *_INPUT_ORDER name lists so the rest of this module can feed/read VMFBs by
# name instead of by position.

def _named_inputs_info(runner: VMFBInferenceRunner, input_order: list, vmfb_path: str) -> dict:
    """Pair ``runner.inputs_info`` (positional, no names) with the hardcoded
    input order. A length mismatch means the model was re-exported with a
    different arity — fail loudly rather than silently feed tensors into the
    wrong argument slots."""
    info = runner.inputs_info  # list[TensorInfo] or None
    if info is None:
        return {}
    if len(info) != len(input_order):
        raise ValueError(
            f"Hardcoded input order ({len(input_order)}) does not match the "
            f"VMFB input count ({len(info)}) for {os.path.basename(vmfb_path)}; the "
            f"model may have been re-exported — update the *_INPUT_ORDER constant in "
            f"runner.py."
        )
    return dict(zip(input_order, info))


def _ordered_feed(feed_dict: dict, input_order: list, input_info: dict) -> list:
    """Build the positional input list a VMFB expects from a name-keyed feed
    dict: cast host arrays to each input's declared dtype, and pass resident
    DeviceArray inputs straight through (no host round-trip, P1/P2)."""
    ordered = []
    for name in input_order:
        val = feed_dict[name]
        if isinstance(val, DeviceArray):
            ordered.append(val)            # resident input (e.g. cross-KV, P1)
            continue
        arr = np.asarray(val).astype(input_info[name].dtype, copy=False)
        ordered.append(arr)
    return ordered


def _to_host_f32(outputs) -> list:
    """Copy VMFB outputs to host as float32 numpy arrays (all Moonshine model
    outputs are floating-point)."""
    result = []
    for o in outputs:
        if hasattr(o, "to_host"):
            o = o.to_host()
        arr = np.asarray(o)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        result.append(arr)
    return result


# ── State ─────────────────────────────────────────────────────────────────────

class MoonshineStaticStreamingState:
    """
    Holds all fixed-size pre-allocated state buffers for static streaming inference.
    Sizes are derived from streaming_config.json and the VMFB model shapes.
    """
    def __init__(self, depth, heads, head_dim, features_dim,
                 conv1_channels, conv2_channels,
                 enc_num_bufs, enc_buf_shape,
                 max_tokens, max_memory_len, total_lookahead):
        self.depth          = depth
        self.heads          = heads
        self.head_dim       = head_dim
        self.features_dim   = features_dim
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.enc_num_bufs   = enc_num_bufs
        self.enc_buf_shape  = enc_buf_shape
        self.max_tokens     = max_tokens
        self.max_memory_len = max_memory_len
        self.total_lookahead = total_lookahead

        # Allocate large fixed buffers once — never reallocated across utterances
        self.k_cross = np.zeros(
            (self.depth, 1, self.heads, self.max_memory_len, self.head_dim), dtype=np.float32
        )
        self.v_cross = np.zeros(
            (self.depth, 1, self.heads, self.max_memory_len, self.head_dim), dtype=np.float32
        )
        self.k_self = np.zeros(
            (self.depth, 1, self.heads, self.max_tokens, self.head_dim), dtype=np.float32
        )
        self.v_self = np.zeros(
            (self.depth, 1, self.heads, self.max_tokens, self.head_dim), dtype=np.float32
        )
        self.pos_offset = np.array([0], dtype=np.int64)

        # P2: resident self-KV device buffers, lazily allocated once by the model
        # and reused thereafter. Deliberately NOT reset between utterances — the
        # original host k_self/v_self are likewise never zeroed; correctness comes
        # from the static decoder overwriting positions 0..step and masking the
        # rest, with committed_tokens governing prefix reuse.
        self.k_self_dev = None
        self.v_self_dev = None

        self.reset()

    def reset(self):
        self.conv1_buffer = np.zeros((1, self.conv1_channels, 4), dtype=np.float32)
        self.conv2_buffer = np.zeros((1, self.conv2_channels, 4), dtype=np.float32)
        self.features_buffer = np.zeros((1, self.total_lookahead, self.features_dim), dtype=np.float32)
        self.enc_bufs     = [np.zeros(self.enc_buf_shape, dtype=np.float32)
                             for _ in range(self.enc_num_bufs)]
        self.pos_offset[0]  = 0
        self.cross_kv_fill  = 0
        self.chunk_idx      = 0

        # Committed-prefix incremental decode state.  self.k_self / self.v_self
        # positions 0..len(committed_tokens)-1 stay valid across previews; only
        # the uncommitted tail is recomputed (see decode_incremental).
        self.committed_tokens = []   # frozen prefix
        self.recent_hyps      = []   # last N hypotheses for LocalAgreement-N


# ── Model ─────────────────────────────────────────────────────────────────────

class MoonshineStaticStreamingModel:
    """
    Orchestrates the 2 VMFB sessions for static streaming inference.

    Args:
        model_dir: Flat directory holding the encoder/decoder ``.vmfb`` files,
            configs, npy tables and tokenizer.
        runtime_flags: Flags forwarded verbatim to the Torq runtime, or ``None``
            to use the runtime's own defaults.
    """
    def __init__(self, model_dir: str, runtime_flags: list[str] | None = None):
        # Flat layout: VMFBs, configs, npy tables and tokenizer all live in
        # model_dir. Input names are hardcoded (see *_INPUT_ORDER above).
        self.model_dir = model_dir

        def load_runner(name, input_order, device_outputs=False):
            vmfb = os.path.join(model_dir, name + ".vmfb")
            runner = VMFBInferenceRunner(
                vmfb,
                device_uri="torq://",
                runtime_flags=runtime_flags,
                device_outputs=device_outputs,
            )
            return runner, _named_inputs_info(runner, input_order, vmfb)

        logger.info("Loading VMFB sessions from %s", model_dir)
        self.fused_encoder, self._enc_info = load_runner(ENCODER_NAME, ENCODER_INPUT_ORDER)
        # P2: device_outputs=True keeps self-KV (and the unread cross-KV/cross_attn)
        # outputs on device; the decode loop copies only logits back to host.
        self.decoder, self._dec_info = load_runner(
            DECODER_NAME, DECODER_INPUT_ORDER, device_outputs=True
        )

        # Load streaming configuration
        cfg_path = find_asset(model_dir, "streaming_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.chunk_len       = cfg["chunk_len"]
        self.feature_stride  = cfg["feature_stride"]
        self.total_lookahead = cfg["total_lookahead"]
        self.warmup_chunks   = cfg["warmup_chunks"]
        self.max_tokens      = cfg["max_tokens"]
        self.max_memory_len  = cfg["max_memory_len"]
        self.extract_embeddings = cfg.get("extract_embeddings", False)

        # Derive model dimensions from fused_encoder inputs
        self.conv1_channels = self._enc_info["conv1_buffer"].shape[1]  # [1, conv1_channels, 4]
        self.conv2_channels = self._enc_info["conv2_buffer"].shape[1]  # [1, conv2_channels, 4]
        self.features_dim   = self._enc_info["features_buffer"].shape[2]  # [1, total_lookahead, features_dim]

        buf_names = sorted(
            (n for n in self._enc_info if n.startswith("buf_")),
            key=lambda n: int(n.split("_")[1])
        )
        self.enc_num_bufs  = len(buf_names)
        self.enc_buf_shape = tuple(self._enc_info[buf_names[0]].shape)

        k_self_0_shape = self._dec_info["k_self_0"].shape
        self.depth    = len([n for n in self._dec_info if n.startswith("k_self_")])
        self.heads    = k_self_0_shape[1]
        self.head_dim = k_self_0_shape[3]

        # Load embedding table when the decoder takes inputs_embeds instead of token ids
        if self.extract_embeddings:
            emb_path = find_asset(model_dir, "decoder_token_embeddings.npy")
            self.token_embeddings = _load_bfloat16_table(emb_path)
        else:
            self.token_embeddings = None

        # Position embedding table for host-side lookup before each adapter call
        pos_emb_path = find_asset(model_dir, "adapter_pos_emb.npy")
        self.pos_emb_weights = _load_bfloat16_table(pos_emb_path)

        logger.debug("Static streaming model specifications (2-Split):")
        logger.debug("  - Depth (Layers):        %d", self.depth)
        logger.debug("  - Attention Heads:       %d", self.heads)
        logger.debug("  - Head Dimension:        %d", self.head_dim)
        logger.debug("  - Features Dimension:    %d", self.features_dim)
        logger.debug("  - Conv1 Channels:        %d", self.conv1_channels)
        logger.debug("  - Conv2 Channels:        %d", self.conv2_channels)
        logger.debug("  - Encoder Bufs:          %d x %s", self.enc_num_bufs, self.enc_buf_shape)
        logger.debug("  - Chunk Length:          %d samples", self.chunk_len)
        logger.debug("  - Feature Stride (F):    %d", self.feature_stride)
        logger.debug("  - Total Lookahead:       %d frames", self.total_lookahead)
        logger.debug("  - Warmup Chunks:         %d", self.warmup_chunks)
        logger.debug("  - Max Tokens:            %d", self.max_tokens)
        logger.debug("  - Max Memory Len:        %d", self.max_memory_len)
        logger.debug("  - Extract Embeddings:    %s", self.extract_embeddings)

    def create_state(self) -> MoonshineStaticStreamingState:
        return MoonshineStaticStreamingState(
            depth=self.depth,
            heads=self.heads,
            head_dim=self.head_dim,
            features_dim=self.features_dim,
            conv1_channels=self.conv1_channels,
            conv2_channels=self.conv2_channels,
            enc_num_bufs=self.enc_num_bufs,
            enc_buf_shape=self.enc_buf_shape,
            max_tokens=self.max_tokens,
            max_memory_len=self.max_memory_len,
            total_lookahead=self.total_lookahead,
        )

    def process_audio_chunk(self, state: MoonshineStaticStreamingState, audio_chunk: np.ndarray):
        """Run the fused encoder and extract Cross-KV updates."""
        F = self.feature_stride
        pos_emb = self.pos_emb_weights[state.pos_offset[0]:state.pos_offset[0] + F].reshape(1, F, -1)

        # Build feed dict
        feed = {
            "audio_chunk":          audio_chunk.reshape(1, -1).astype(np.float32),
            "conv1_buffer":         state.conv1_buffer,
            "conv2_buffer":         state.conv2_buffer,
            "features_buffer":      state.features_buffer,
            "position_embeddings":  pos_emb,
        }
        for i, buf in enumerate(state.enc_bufs):
            feed[f"buf_{i}"] = buf

        res = _to_host_f32(self.fused_encoder.infer(
            _ordered_feed(feed, ENCODER_INPUT_ORDER, self._enc_info)
        ))

        # Unpack outputs:
        # k_cross, v_cross, conv1_buffer_out, conv2_buffer_out, features_buffer_out, *buf_out
        new_k, new_v = res[0], res[1]
        state.conv1_buffer = res[2]
        state.conv2_buffer = res[3]
        state.features_buffer = res[4]

        # Warmup vs Active step
        if state.chunk_idx < self.warmup_chunks:
            # Warmup: discard outputs and encoder buffer updates
            pass
        else:
            # Active step
            for i in range(self.enc_num_bufs):
                state.enc_bufs[i] = res[5 + i]

            # Save cross-KV
            end = min(state.cross_kv_fill + F, self.max_memory_len)
            take = end - state.cross_kv_fill
            state.k_cross[:, :, :, state.cross_kv_fill:end, :] = new_k[:, :, :, :take, :]
            state.v_cross[:, :, :, state.cross_kv_fill:end, :] = new_v[:, :, :, :take, :]
            state.cross_kv_fill = end

            state.pos_offset[0] += F

        state.chunk_idx += 1

    def encode(self, state: MoonshineStaticStreamingState, is_final: bool):
        """
        No-op during streaming (logic fully embedded in process_audio_chunk).
        On final flush: feeds silent chunks through process_audio_chunk to push
        out the remaining lookahead frames (chunk_idx is already past warmup by
        this point, so each zero chunk takes the same "active" path as real audio).
        """
        if not is_final:
            return
        zero_chunk = np.zeros(self.chunk_len, dtype=np.float32)
        for _ in range(self.warmup_chunks):
            self.process_audio_chunk(state, zero_chunk)

    def _upload_cross_kv(self, state: MoonshineStaticStreamingState):
        """P1: upload the currently-valid cross-KV to resident device buffers once
        per decode call (cast to the decoder's input dtype), so the per-token loop
        reuses the same handles instead of re-uploading the full cross-KV
        (~88 % of per-token H2D) every step. Cross-KV is constant within a single
        decode call. Falls back to host arrays if the session lacks device support."""
        if not hasattr(self.decoder, "allocate_device_array"):
            return ([state.k_cross[i] for i in range(self.depth)],
                    [state.v_cross[i] for i in range(self.depth)])
        dt = self._dec_info["k_cross_0"].dtype

        def up(x):
            return self.decoder.allocate_device_array(x.astype(dt, copy=False))

        return ([up(state.k_cross[i]) for i in range(self.depth)],
                [up(state.v_cross[i]) for i in range(self.depth)])

    def _ensure_self_kv_device(self, state: MoonshineStaticStreamingState) -> bool:
        """P2: lazily allocate the resident self-KV device buffers (once). Returns
        True when self-KV is device-resident (decode loop feeds/writes DeviceArrays
        and skips the per-token self-KV host round-trip), False to fall back to the
        host numpy buffers. Not re-zeroed per utterance — see State for why."""
        if not hasattr(self.decoder, "allocate_device_array"):
            return False
        if state.k_self_dev is not None:
            return True
        dt = self._dec_info["k_self_0"].dtype

        def up(x):
            return self.decoder.allocate_device_array(x.astype(dt, copy=False))

        state.k_self_dev = [up(state.k_self[i]) for i in range(self.depth)]
        state.v_self_dev = [up(state.v_self[i]) for i in range(self.depth)]
        return True

    def _decoder_step(self, state, first_feed, cross_attn_bias, current_len,
                      k_cross_dev, v_cross_dev, use_dev) -> int:
        """One decoder forward pass. Self-KV and cross-KV are fed as resident
        device buffers (P1/P2); only logits are copied to host (for argmax). When
        use_dev, the self-KV outputs stay on device and become the next step's
        inputs — no per-token self-KV round-trip."""
        dec_feed = {
            **first_feed,
            "cross_attn_bias": cross_attn_bias,
            "current_len":     current_len,   # ignored by VMFB (not in model inputs)
            "position_ids":    current_len,
        }
        for _i in range(self.depth):
            if use_dev:
                dec_feed[f"k_self_{_i}"] = state.k_self_dev[_i]
                dec_feed[f"v_self_{_i}"] = state.v_self_dev[_i]
            else:
                dec_feed[f"k_self_{_i}"] = state.k_self[_i]
                dec_feed[f"v_self_{_i}"] = state.v_self[_i]
            dec_feed[f"k_cross_{_i}"] = k_cross_dev[_i]
            dec_feed[f"v_cross_{_i}"] = v_cross_dev[_i]

        dec_out = self.decoder.infer(
            _ordered_feed(dec_feed, DECODER_INPUT_ORDER, self._dec_info)
        )

        # Only logits cross back to host (for argmax); match the original f32 path.
        logits = dec_out[0]
        if hasattr(logits, "to_host"):
            logits = logits.to_host()
        logits = np.asarray(logits).astype(np.float32, copy=False)

        # Self-KV outputs: keep resident (next-step inputs) or write back to host.
        if use_dev:
            for _i in range(self.depth):
                state.k_self_dev[_i] = dec_out[1 + _i * 2]
                state.v_self_dev[_i] = dec_out[2 + _i * 2]
        else:
            for _i in range(self.depth):
                ko, vo = dec_out[1 + _i * 2], dec_out[2 + _i * 2]
                state.k_self[_i] = np.asarray(ko.to_host() if hasattr(ko, "to_host") else ko)
                state.v_self[_i] = np.asarray(vo.to_host() if hasattr(vo, "to_host") else vo)

        return int(np.argmax(logits[0, 0, :]))

    def _decode_loop(self, state, start_step, start_token, max_tokens,
                      cross_attn_bias, k_cross_dev, v_cross_dev, use_dev):
        """Greedy-decode one token at a time from ``start_step``/``start_token``
        until EOS (token=2) or max_tokens. Returns new tokens; shared
        by decode() (start_step=0, start_token=BOS) and decode_incremental()
        (resuming after the committed prefix)."""
        tokens        = []
        current_token = start_token
        step          = start_step

        while step < max_tokens:
            current_len = np.array([[step]], dtype=np.int64)

            if self.extract_embeddings:
                first_feed = {"inputs_embeds": self.token_embeddings[current_token].reshape(1, 1, -1)}
            else:
                first_feed = {"token": np.array([[current_token]], dtype=np.int64)}

            next_token = self._decoder_step(
                state, first_feed, cross_attn_bias, current_len,
                k_cross_dev, v_cross_dev, use_dev,
            )
            step      += 1

            if next_token == 2 or step >= max_tokens:
                break

            tokens.append(next_token)
            current_token = next_token

        return tokens

    def decode(self, state: MoonshineStaticStreamingState):
        """
        Autoregressive token generation using pre-allocated static KV buffers.
        Starts from BOS (token=1) and generates until EOS (token=2) or max_tokens.
        Returns the list of generated token IDs (excluding BOS/EOS).
        """
        if state.cross_kv_fill == 0:
            return []

        duration_sec = state.cross_kv_fill * 0.020
        max_tokens   = min(int(math.ceil(duration_sec * 6.5)), self.max_tokens)

        cross_attn_bias = np.zeros((1, self.heads, 1, self.max_memory_len), dtype=np.float32)
        cross_attn_bias[:, :, :, state.cross_kv_fill:] = -1e9

        # P1: cross-KV is constant for this whole decode call — upload once.
        k_cross_dev, v_cross_dev = self._upload_cross_kv(state)
        # P2: ensure self-KV resides on device (allocated once, then reused).
        use_dev = self._ensure_self_kv_device(state)

        tokens = self._decode_loop(
            state, 0, 1, max_tokens, cross_attn_bias, k_cross_dev, v_cross_dev, use_dev
        )
        return tokens

    def decode_incremental(self, state: MoonshineStaticStreamingState,
                           commit_delay_sec: float = 3.0, agreement: int = 2):
        """
        Committed-prefix incremental decode (O(tail) instead of O(T) from BOS).

        Resumes from the committed prefix: self-KV positions 0..C-1 are kept
        intact and only the uncommitted tail is regenerated.  After decoding,
        the committed prefix is advanced to tokens that are BOTH
        LocalAgreement-`agreement` stable AND at least `commit_delay_sec` of
        audio behind the live frontier.

        Returns the full hypothesis (committed prefix + freshly decoded tail).
        """
        if state.cross_kv_fill == 0:
            return state.committed_tokens[:]

        duration_sec = state.cross_kv_fill * 0.020
        max_tokens   = min(int(math.ceil(duration_sec * 6.5)), self.max_tokens)

        cross_attn_bias = np.zeros((1, self.heads, 1, self.max_memory_len), dtype=np.float32)
        cross_attn_bias[:, :, :, state.cross_kv_fill:] = -1e9

        # P1: cross-KV is constant for this whole decode call — upload once.
        k_cross_dev, v_cross_dev = self._upload_cross_kv(state)
        # P2: ensure self-KV resides on device (allocated once, then reused). The
        # committed prefix's self-KV persists on device across preview decodes.
        use_dev = self._ensure_self_kv_device(state)

        # Resume from the committed prefix (self-KV positions 0..C-1 are valid).
        committed     = state.committed_tokens
        C             = len(committed)
        start_token   = committed[-1] if C else 1  # last committed token, re-fed at position C (or BOS)

        tail_tokens = self._decode_loop(
            state, C, start_token, max_tokens, cross_attn_bias, k_cross_dev, v_cross_dev, use_dev
        )
        result_tokens = committed + tail_tokens

        # ── Commit rule: LocalAgreement-N  AND  ≥ commit_delay_sec behind frontier ──
        state.recent_hyps.append(result_tokens[:])
        if len(state.recent_hyps) > agreement:
            state.recent_hyps.pop(0)

        if len(state.recent_hyps) >= agreement:
            la_len = _agree_prefix_len(state.recent_hyps)     # LocalAgreement-N prefix
        else:
            la_len = C                                        # not enough history yet

        # Min-age gate: uniform token→audio alignment; commit only tokens whose
        # estimated audio position is >= commit_delay_sec behind the frontier.
        T            = len(result_tokens)
        fill         = state.cross_kv_fill
        delay_frames = commit_delay_sec / 0.020               # 20 ms per cross-KV frame
        if T > 0 and fill > 0:
            frac_old  = max(0.0, (fill - delay_frames) / fill)
            age_len   = int(T * frac_old)
        else:
            age_len   = 0

        commit_len = max(C, min(la_len, age_len))             # monotonic, never un-commit
        state.committed_tokens = result_tokens[:commit_len]
        return result_tokens
