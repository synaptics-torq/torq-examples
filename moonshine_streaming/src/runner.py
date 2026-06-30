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
from types import SimpleNamespace

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
# re-export with a different arity trips the arity check in ``_Session``; a pure
# reordering would not, so keep these in sync with the model if it is rebuilt.
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


# ── Thin VMFB session wrapper ─────────────────────────────────────────────────

class _Session:
    """
    Wraps VMFBInferenceRunner with a dict-based run() interface similar to ORT.

    Input names come from the hardcoded ``input_order`` (the VMFB exposes inputs
    positionally, with no names); their shapes and dtypes come from
    ``VMFBInferenceRunner.inputs_info``.  Outputs are always returned as float32
    numpy arrays (all Moonshine model outputs are floating-point).
    """

    def __init__(self, vmfb_path: str, input_order: list, runtime_flags: list,
                 device_outputs: bool = False, function: str = "main"):
        self._runner = VMFBInferenceRunner(
            vmfb_path,
            device_uri="torq://",
            function=function,
            runtime_flags=runtime_flags,
            device_outputs=device_outputs,
        )
        self._input_names = list(input_order)
        # The VMFB reports shapes + dtypes positionally (no names) via inputs_info;
        # pair them with the hardcoded input order. A length mismatch means the
        # model was re-exported with a different arity — fail loudly rather than
        # silently feed tensors into the wrong argument slots.
        info = self._runner.inputs_info  # list[TensorInfo] or None
        if info is not None and len(info) != len(self._input_names):
            raise ValueError(
                f"Hardcoded input order ({len(self._input_names)}) does not match the "
                f"VMFB input count ({len(info)}) for {os.path.basename(vmfb_path)}; the "
                f"model may have been re-exported — update the *_INPUT_ORDER constant in "
                f"runner.py."
            )
        self._dtypes       = info
        self._input_shapes = (
            [list(t.shape) for t in info] if info is not None
            else [None] * len(self._input_names)
        )
        # name -> model input dtype, for pre-uploading resident device buffers (P1)
        self.input_dtype = (
            {n: t.dtype for n, t in zip(self._input_names, info)}
            if info else {}
        )

    def get_inputs(self) -> list:
        """Return ordered list of SimpleNamespace(name, shape) — mirrors ORT API."""
        return [
            SimpleNamespace(name=n, shape=s)
            for n, s in zip(self._input_names, self._input_shapes)
        ]

    def allocate_device_array(self, arr) -> DeviceArray:
        """Upload a host array to a resident device buffer (caller must cast to
        the model's input dtype first). The returned DeviceArray can be fed to
        run() repeatedly without re-uploading (P1)."""
        return self._runner.allocate_device_array(arr)

    def _ordered_inputs(self, feed_dict: dict) -> list:
        """Build the ordered input list: cast host arrays to the model dtype, and
        pass resident DeviceArray inputs straight through (no host round-trip)."""
        ordered = []
        for i, name in enumerate(self._input_names):
            val = feed_dict[name]
            if isinstance(val, DeviceArray):
                ordered.append(val)            # resident input (e.g. cross-KV, P1)
                continue
            arr = np.asarray(val)
            if self._dtypes and i < len(self._dtypes):
                arr = arr.astype(self._dtypes[i].dtype, copy=False)
            ordered.append(arr)
        return ordered

    def run(self, feed_dict: dict) -> list:
        """Run inference; return all outputs as float32 numpy arrays."""
        raw = self._runner.infer(self._ordered_inputs(feed_dict))
        result = []
        for o in raw:
            if hasattr(o, "to_host"):
                o = o.to_host()
            arr = np.asarray(o)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            result.append(arr)
        return result

    def run_raw(self, feed_dict: dict) -> list:
        """Run inference; return the runner's raw outputs WITHOUT copying to host.
        With device_outputs=True these are DeviceArrays, letting the caller keep
        self-KV resident and to_host only the logits (P2)."""
        return self._runner.infer(self._ordered_inputs(feed_dict))


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
        self.last_decode_steps = 0   # decoder forward passes in the most recent decode()

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
        hw_type:   Torq hardware type flag (e.g. ``astra_machina``, ``sim``).
        function:  VMFB entry function name.
    """
    def __init__(self, model_dir: str, hw_type: str, function: str = "main"):
        runtime_flags = [f"--torq_hw_type={hw_type}"]

        # Flat layout: VMFBs, configs, npy tables and tokenizer all live in
        # model_dir. Input names are hardcoded (see *_INPUT_ORDER above).
        self.model_dir = model_dir

        def session(name, input_order, device_outputs=False):
            vmfb = os.path.join(model_dir, name + ".vmfb")
            return _Session(vmfb, input_order, runtime_flags,
                            device_outputs=device_outputs, function=function)

        logger.info("Loading VMFB sessions from %s", model_dir)
        self.fused_encoder = session(ENCODER_NAME, ENCODER_INPUT_ORDER)
        # P2: device_outputs=True keeps self-KV (and the unread cross-KV/cross_attn)
        # outputs on device; the decode loop copies only logits back to host.
        self.decoder = session(DECODER_NAME, DECODER_INPUT_ORDER, device_outputs=True)

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
        fe_inputs = self.fused_encoder.get_inputs()
        self.conv1_channels = fe_inputs[1].shape[1]  # conv1_buffer: [1, conv1_channels, 4]
        self.conv2_channels = fe_inputs[2].shape[1]  # conv2_buffer: [1, conv2_channels, 4]
        self.features_dim = fe_inputs[3].shape[2]   # features_buffer: [1, total_lookahead, features_dim]

        buf_inputs = sorted(
            [i for i in fe_inputs if i.name.startswith("buf_")],
            key=lambda x: int(x.name.split("_")[1])
        )
        self.enc_num_bufs  = len(buf_inputs)
        self.enc_buf_shape = tuple(buf_inputs[0].shape)

        dec_inputs = self.decoder.get_inputs()
        k_self_0_shape  = next(i for i in dec_inputs if i.name == "k_self_0").shape
        self.depth    = len([i for i in dec_inputs if i.name.startswith("k_self_")])
        self.heads    = k_self_0_shape[1]
        self.head_dim = k_self_0_shape[3]

        # Load embedding table when the decoder takes inputs_embeds instead of token ids
        if self.extract_embeddings:
            emb_path = find_asset(model_dir, "decoder_token_embeddings.npy")
            self.token_embeddings = np.load(emb_path).astype(np.float32)
        else:
            self.token_embeddings = None

        # Position embedding table for host-side lookup before each adapter call
        pos_emb_path = find_asset(model_dir, "adapter_pos_emb.npy")
        self.pos_emb_weights = np.load(pos_emb_path).astype(np.float32)

        logger.info("Static streaming model specifications (2-Split):")
        logger.info("  - Depth (Layers):        %d", self.depth)
        logger.info("  - Attention Heads:       %d", self.heads)
        logger.info("  - Head Dimension:        %d", self.head_dim)
        logger.info("  - Features Dimension:    %d", self.features_dim)
        logger.info("  - Conv1 Channels:        %d", self.conv1_channels)
        logger.info("  - Conv2 Channels:        %d", self.conv2_channels)
        logger.info("  - Encoder Bufs:          %d x %s", self.enc_num_bufs, self.enc_buf_shape)
        logger.info("  - Chunk Length:          %d samples", self.chunk_len)
        logger.info("  - Feature Stride (F):    %d", self.feature_stride)
        logger.info("  - Total Lookahead:       %d frames", self.total_lookahead)
        logger.info("  - Warmup Chunks:         %d", self.warmup_chunks)
        logger.info("  - Max Tokens:            %d", self.max_tokens)
        logger.info("  - Max Memory Len:        %d", self.max_memory_len)
        logger.info("  - Extract Embeddings:    %s", self.extract_embeddings)

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

        res = self.fused_encoder.run(feed)

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
        On final flush: feeds silent chunks to push out remaining lookahead frames.
        """
        if not is_final:
            return

        # Flush final lookahead frames by feeding zero chunks
        F = self.feature_stride
        for _ in range(self.warmup_chunks):
            zero_chunk = np.zeros((1, self.chunk_len), dtype=np.float32)
            pos_emb = self.pos_emb_weights[state.pos_offset[0]:state.pos_offset[0] + F].reshape(1, F, -1)

            feed = {
                "audio_chunk":          zero_chunk,
                "conv1_buffer":         state.conv1_buffer,
                "conv2_buffer":         state.conv2_buffer,
                "features_buffer":      state.features_buffer,
                "position_embeddings":  pos_emb,
            }
            for i, buf in enumerate(state.enc_bufs):
                feed[f"buf_{i}"] = buf

            res = self.fused_encoder.run(feed)

            new_k, new_v = res[0], res[1]
            state.conv1_buffer = res[2]
            state.conv2_buffer = res[3]
            state.features_buffer = res[4]

            # Since chunk_idx >= warmup_chunks during final flush, always update caches and save cross-KV
            for i in range(self.enc_num_bufs):
                state.enc_bufs[i] = res[5 + i]

            end = min(state.cross_kv_fill + F, self.max_memory_len)
            take = end - state.cross_kv_fill
            state.k_cross[:, :, :, state.cross_kv_fill:end, :] = new_k[:, :, :, :take, :]
            state.v_cross[:, :, :, state.cross_kv_fill:end, :] = new_v[:, :, :, :take, :]
            state.cross_kv_fill = end

            state.pos_offset[0] += F

    def _upload_cross_kv(self, state: MoonshineStaticStreamingState):
        """P1: upload the currently-valid cross-KV to resident device buffers once
        per decode call (cast to the decoder's input dtype), so the per-token loop
        reuses the same handles instead of re-uploading the full cross-KV
        (~88 % of per-token H2D) every step. Cross-KV is constant within a single
        decode call. Falls back to host arrays if the session lacks device support."""
        if not hasattr(self.decoder, "allocate_device_array"):
            return ([state.k_cross[i] for i in range(self.depth)],
                    [state.v_cross[i] for i in range(self.depth)])
        dt = self.decoder.input_dtype.get("k_cross_0")

        def up(x):
            if dt is not None:
                x = x.astype(dt, copy=False)
            return self.decoder.allocate_device_array(x)

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
        dt = self.decoder.input_dtype.get("k_self_0")

        def up(x):
            if dt is not None:
                x = x.astype(dt, copy=False)
            return self.decoder.allocate_device_array(x)

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

        dec_out = self.decoder.run_raw(dec_feed)

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

    def decode(self, state: MoonshineStaticStreamingState):
        """
        Autoregressive token generation using pre-allocated static KV buffers.
        Starts from BOS (token=1) and generates until EOS (token=2) or max_tokens.
        Returns the list of generated token IDs (excluding BOS/EOS).
        """
        if state.cross_kv_fill == 0:
            state.last_decode_steps = 0
            return []

        duration_sec = state.cross_kv_fill * 0.020
        max_tokens   = min(int(math.ceil(duration_sec * 6.5)), self.max_tokens)

        cross_attn_bias = np.zeros((1, self.heads, 1, self.max_memory_len), dtype=np.float32)
        cross_attn_bias[:, :, :, state.cross_kv_fill:] = -1e9

        # P1: cross-KV is constant for this whole decode call — upload once.
        k_cross_dev, v_cross_dev = self._upload_cross_kv(state)
        # P2: ensure self-KV resides on device (allocated once, then reused).
        use_dev = self._ensure_self_kv_device(state)

        result_tokens = []
        current_token = 1  # BOS
        step = 0

        while True:
            current_len = np.array([[step]], dtype=np.int64)

            if self.extract_embeddings:
                first_feed = {"inputs_embeds": self.token_embeddings[current_token].reshape(1, 1, -1)}
            else:
                first_feed = {"token": np.array([[current_token]], dtype=np.int64)}

            next_token = self._decoder_step(
                state, first_feed, cross_attn_bias, current_len,
                k_cross_dev, v_cross_dev, use_dev,
            )
            step += 1

            if next_token == 2 or step >= max_tokens:
                break

            result_tokens.append(next_token)
            current_token = next_token

        state.last_decode_steps = step
        return result_tokens

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
            state.last_decode_steps = 0
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
        result_tokens = committed[:]
        if C == 0:
            current_token = 1   # BOS
            step          = 0
        else:
            current_token = committed[-1]  # last committed token, re-fed at position C
            step          = C

        steps_run = 0
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
            steps_run += 1

            if next_token == 2 or step >= max_tokens:
                break

            result_tokens.append(next_token)
            current_token = next_token

        state.last_decode_steps = steps_run

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
