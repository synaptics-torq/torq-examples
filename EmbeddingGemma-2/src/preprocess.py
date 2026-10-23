# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Host-side EmbeddingGemma-2 pre/post-processing (numpy + PIL only).

Mirrors the HF ``Gemma4ImageProcessor`` / ``EmbeddingGemma2VideoProcessor`` and the
``EmbeddingGemma2Processor`` token layout, so the same code can run on the board without
torch or transformers.

Token layout (``tokenizer.json`` post-processor adds BOS/EOS):
    text   : <bos> prompt+text <eos>
    image  : <bos> <|image> <|image|>*n <image|> <eos>
    video  : <bos> (<|image> <|video|>*n <image|>) * frames <eos>
    audio  : <bos> <|audio> <|audio|>*n <audio|> <eos>
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

PATCH_SIZE = 16
POOL_KERNEL = 3
SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)

BOS_ID = 2
EOS_ID = 1
PAD_ID = 0
BOI_ID = 255999
EOI_ID = 258882
IMAGE_TOKEN_ID = 258880
VIDEO_TOKEN_ID = 258884
BOA_ID = 256000
EOA_ID = 258883
AUDIO_TOKEN_ID = 258881

# Gemma4 audio front end: 16 kHz mono, 20 ms frames, 10 ms hop, 512-point FFT, 128 mel
# bins, two stride-2 subsampling convs -> one token per 40 ms.
AUDIO_SAMPLE_RATE = 16000
AUDIO_FRAME_LENGTH = 320
AUDIO_HOP_LENGTH = 160
AUDIO_FFT_LENGTH = 512
AUDIO_MEL_FLOOR = 1e-3
AUDIO_PAD_MULTIPLE = 128

PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "SearchQuery": "task: search result | query: ",
    "Document": "title: none | text: ",
    "QuestionAnswering": "task: question answering | query: ",
    "FactChecking": "task: fact checking | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "STS": "task: sentence similarity | query: ",
}

MRL_DIMS = (768, 512, 256, 128)
MASK_BIAS = -1e9  # additive attention bias for padded keys


def aspect_preserving_size(height: int, width: int, max_soft_tokens: int) -> tuple[int, int]:
    """Target (H, W) of HF ``get_aspect_ratio_preserving_size`` for a soft-token budget."""
    max_patches = max_soft_tokens * POOL_KERNEL**2
    target_px = max_patches * PATCH_SIZE**2
    factor = math.sqrt(target_px / (height * width))
    side = POOL_KERNEL * PATCH_SIZE
    th = int(math.floor(factor * height / side)) * side
    tw = int(math.floor(factor * width / side)) * side
    max_side = (max_patches // POOL_KERNEL**2) * side
    if th == 0 and tw == 0:
        raise ValueError(f"cannot resize {height}x{width} to a non-empty grid")
    if th == 0:
        th, tw = side, min(int(math.floor(width / height)) * side, max_side)
    elif tw == 0:
        tw, th = side, min(int(math.floor(height / width)) * side, max_side)
    return th, tw


@dataclass
class VisionInput:
    patches: np.ndarray  # [N, 768] float32 in [0, 1]
    positions: np.ndarray  # [N, 2] int64, (x, y)
    grid: tuple[int, int]  # patch grid (rows, cols)

    @property
    def num_soft_tokens(self) -> int:
        return self.patches.shape[0] // POOL_KERNEL**2


def patchify(image: Image.Image, size: tuple[int, int]) -> VisionInput:
    """Resize (bicubic) to ``size`` = (H, W), rescale to [0, 1] and cut 16x16 patches.

    Patch layout follows ``convert_image_to_patches``: each row is (py, px, c) flattened,
    patches are row-major over the grid, positions are (x, y).
    """
    h, w = size
    img = image.convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)  # [H, W, C]
    gh, gw = h // PATCH_SIZE, w // PATCH_SIZE
    patches = arr.reshape(gh, PATCH_SIZE, gw, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, -1)
    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    positions = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1).astype(np.int64)
    return VisionInput(np.ascontiguousarray(patches), positions, (gh, gw))


def preprocess_image(image: Image.Image, mode: str = "square", max_soft_tokens: int = 70,
                     square_size: int = 384) -> VisionInput:
    """``mode='square'``: fixed square resize (the static Torq geometry).
    ``mode='aspect'``: HF aspect-preserving resize for ``max_soft_tokens``."""
    if mode == "square":
        return patchify(image, (square_size, square_size))
    if mode == "aspect":
        if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}")
        return patchify(image, aspect_preserving_size(image.height, image.width, max_soft_tokens))
    raise ValueError(f"unknown mode {mode!r}")


def sample_video_frames(path: str, fps: float = 1.0, max_frames: int = 7, return_times: bool = False):
    """Sample frames at ``fps`` (uniform over the clip); if more than ``max_frames``,
    keep ``max_frames`` uniformly spaced ones. Mirrors the HF ``fps`` + ``uniform`` overflow.
    With ``return_times``, also returns each frame's time in seconds."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration = total / src_fps if src_fps > 0 else 0.0
    n = max(1, int(duration * fps)) if duration > 0 else total
    idx = np.linspace(0, total - 1, num=min(n, total)).round().astype(int)
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, num=max_frames).round().astype(int)]
    # Grab sequentially and decode only the selected frames: seeking re-decodes
    # from the previous keyframe for every sample.
    wanted = set(int(i) for i in idx)
    frames, times = [], []
    for i in range(int(idx[-1]) + 1):
        if not cap.grab():
            break
        if i in wanted:
            ok, bgr = cap.retrieve()
            if ok:
                frames.append(Image.fromarray(bgr[:, :, ::-1].copy()))
                times.append(i / src_fps if src_fps > 0 else 0.0)
    cap.release()
    return (frames, times) if return_times else frames


def load_audio(path: str) -> np.ndarray:
    """PCM WAV -> float32 mono at 16 kHz (channels averaged, FFT resampling)."""
    import wave

    with wave.open(str(path)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    x = (x - 128.0) / 128.0 if width == 1 else x / float(np.iinfo(dtype).max + 1)
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != AUDIO_SAMPLE_RATE:
        n = int(round(len(x) * AUDIO_SAMPLE_RATE / sr))
        spec = np.fft.rfft(x)
        out = np.zeros(n // 2 + 1, dtype=np.complex128)
        k = min(len(spec), len(out))
        out[:k] = spec[:k]
        x = (np.fft.irfft(out, n=n) * (n / len(x))).astype(np.float32)
    return x


def audio_features(waveform: np.ndarray, mel_filters: np.ndarray, window: np.ndarray,
                   num_frames: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Log-mel features [T, 128] and frame mask [T] (port of Gemma4AudioFeatureExtractor).

    The waveform is right-padded to a multiple of 128 samples (as the HF extractor does);
    with ``num_frames`` the result is zero-padded / truncated to that many frames."""
    x = np.asarray(waveform, dtype=np.float32)
    valid = np.ones(len(x), dtype=bool)
    pad = (-len(x)) % AUDIO_PAD_MULTIPLE
    x = np.pad(x, (0, pad)); valid = np.pad(valid, (0, pad))
    left = AUDIO_FRAME_LENGTH // 2
    x = np.pad(x, (left, 0)); valid = np.pad(valid, (left, 0))
    size = AUDIO_FRAME_LENGTH + 1
    n = (len(x) - size) // AUDIO_HOP_LENGTH + 1
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, size), strides=(x.strides[0] * AUDIO_HOP_LENGTH, x.strides[0]))[:, :-1]
    spec = np.abs(np.fft.rfft(frames * window, n=AUDIO_FFT_LENGTH, axis=-1))
    mel = np.log(spec @ mel_filters + AUDIO_MEL_FLOOR).astype(np.float32)
    mask = valid[np.arange(n) * AUDIO_HOP_LENGTH + size - 1]
    mel = mel * mask[:, None]
    if num_frames is not None:
        mel = np.pad(mel, ((0, max(0, num_frames - n)), (0, 0)))[:num_frames]
        mask = np.pad(mask, (0, max(0, num_frames - n)))[:num_frames]
    return mel, mask


def audio_num_tokens(frame_mask: np.ndarray) -> int:
    """Valid audio tokens after the two stride-2 subsampling convs (processor arithmetic)."""
    m = np.asarray(frame_mask, dtype=bool)
    for _ in range(2):
        t = (len(m) + 2 - 3) // 2 + 1
        m = m[::2][:t]
    return int(m.sum())


def audio_ids(num_tokens: int) -> list[int]:
    return [BOS_ID, BOA_ID] + [AUDIO_TOKEN_ID] * num_tokens + [EOA_ID, EOS_ID]


def text_ids(tokenizer, text: str, prompt_name: str | None = None) -> list[int]:
    prefix = PROMPTS[prompt_name] if prompt_name else ""
    return tokenizer.encode(prefix + text).ids


def image_ids(num_soft_tokens: int) -> list[int]:
    return [BOS_ID, BOI_ID] + [IMAGE_TOKEN_ID] * num_soft_tokens + [EOI_ID, EOS_ID]


def video_ids(num_frames: int, num_soft_tokens: int) -> list[int]:
    frame = [BOI_ID] + [VIDEO_TOKEN_ID] * num_soft_tokens + [EOI_ID]
    return [BOS_ID] + frame * num_frames + [EOS_ID]


def build_inputs_embeds(ids: list[int], lut: np.ndarray, seq_len: int,
                        media: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gather (pre-scaled) token embeddings, write ``media`` rows [M, 512] into the
    image/video placeholder positions in order, and right-pad to ``seq_len``.

    Returns (inputs_embeds [1, S, D] float32, attention_mask [1, S] int64)."""
    if len(ids) > seq_len:
        raise ValueError(f"{len(ids)} tokens exceed the static sequence length {seq_len}")
    ids_arr = np.asarray(ids, dtype=np.int64)
    rows = np.asarray(lut[ids_arr], dtype=np.float32)
    slots = np.flatnonzero((ids_arr == IMAGE_TOKEN_ID) | (ids_arr == VIDEO_TOKEN_ID)
                           | (ids_arr == AUDIO_TOKEN_ID))
    if media is not None or len(slots):
        if media is None or media.shape[0] != len(slots):
            raise ValueError(f"{len(slots)} media placeholders but "
                             f"{0 if media is None else media.shape[0]} media rows")
        rows[slots] = media
    embeds = np.zeros((1, seq_len, lut.shape[1]), dtype=np.float32)
    embeds[0, : len(ids)] = rows
    mask = np.zeros((1, seq_len), dtype=np.int64)
    mask[0, : len(ids)] = 1
    return embeds, mask


def attention_bias(attention_mask: np.ndarray) -> np.ndarray:
    """[1, S] 0/1 mask -> [1, 1, 1, S] additive key bias (0 for real tokens)."""
    m = np.asarray(attention_mask)
    return np.where(m[:, None, None, :] > 0, 0.0, MASK_BIAS).astype(np.float32)


def pool_and_normalize(last_hidden_state: np.ndarray, attention_mask: np.ndarray,
                       dim: int = 768) -> np.ndarray:
    """Masked mean over real tokens, optional MRL truncation, then L2 normalisation."""
    h = np.asarray(last_hidden_state, dtype=np.float32)[0]
    m = np.asarray(attention_mask, dtype=np.float32)[0, : h.shape[0], None]
    pooled = (h * m).sum(0) / max(m.sum(), 1.0)
    pooled = pooled[:dim]
    return pooled / max(np.linalg.norm(pooled), 1e-12)
