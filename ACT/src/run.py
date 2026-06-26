# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Run the ACT policy on the board, end to end, from one camera frame + robot state to a
100-step action chunk:

    backbone_fused.vmfb   int16 image [1,480,640,3] -> bf16 image features     NSS
    [numpy host glue]     state projection + token concat + positional embedding
    enc4_dattn.vmfb       4 encoder layers + decoder cross-attention            NSS
    decoder_ffn.vmfb      decoder FFN + action head -> action                   NSS
    denormalize           action * a_std + a_mean -> physical joint targets

All artifacts are read from a model directory laid out by `setup_demos.py ACT`
(default: models/<repo_id>/): the three .vmfb modules, `glue_params.npz` (pre-encoder host-glue
constants + action denorm stats), and an optional bundled sample frame. The host-glue arithmetic is
reproduced here in pure numpy (verified bit-for-bit against the original onnx subgraph), so the board
needs only torq-runtime, numpy, and ml_dtypes.

Run from the ACT directory:

    cd ACT
    python src/run.py                                  # uses the bundled sample frame
    python src/run.py -m ../models/Synaptics/act-so101-torq    # explicit model dir
    python src/run.py --image img.bin --state state.bin        # your own pre-normalized inputs

Input formats (raw .bin, C-order): image = int16 NHWC [1,480,640,3]
(round(normalized_image / stem_in_scale)); state = float32 [1,D] ((state-mean)/std).
"""

import argparse
import logging
import time
from pathlib import Path

import ml_dtypes
import numpy as np
from torq.runtime import VMFBInferenceRunner

from ACT.setup_demo import DEFAULT_REPO, ensure_act_models
from utils.download import default_models_dir
from utils.log import add_logging_args, configure_logging

logger = logging.getLogger("ACT")

H = 100                       # ACT action-chunk horizon
bf16 = ml_dtypes.bfloat16


def _to_bf16(a):
    return np.ascontiguousarray(a).astype(bf16)          # f32 -> bf16 (round to nearest)


def _to_f32(a):
    return np.ascontiguousarray(a).astype(np.float32)    # bf16/any -> f32


def host_glue(permute1_f32, state_f32, g):
    """permute_1 (image features) + state -> add_50, stack_2  (each [302,1,512]).

    Reproduces the pre-encoder prefix of the transformer exactly in numpy:
      state token  = state @ Ws^T + bs                       (robot-state input projection)
      latent token = g['latent']                             (latent input proj of zeros == its bias)
      image tokens = permute_1 reshaped to [300,1,512]
      stack_2      = concat([latent, state, image], axis=0)  (token VALUES)            [302,1,512]
      add_50       = stack_2 + g['pos']                      (values + positional emb) [302,1,512]
    g['pos'] is the fixed sin/cos positional embedding (input-independent).
    """
    Ws, bs, latent, pos = g["Ws"], g["bs"], g["latent"], g["pos"]
    state_tok = (state_f32 @ Ws.T + bs).reshape(1, 1, 512)
    latent_tok = latent.reshape(1, 1, 512)
    image_tok = permute1_f32.reshape(300, 1, 512)
    stack_2 = np.concatenate([latent_tok, state_tok, image_tok], axis=0)   # [302,1,512]
    add_50 = stack_2 + pos                                                 # [302,1,512]
    return add_50, stack_2


def main(args: argparse.Namespace) -> None:
    configure_logging(args.logging)
    logger.info("Starting ACT policy...")

    model_dir = Path(args.model) if args.model else default_models_dir() / DEFAULT_REPO
    ensure_act_models(model_dir, refresh=not args.no_refresh)

    for name in ("backbone_fused.vmfb", "enc4_dattn.vmfb", "decoder_ffn.vmfb", "glue_params.npz"):
        if not (model_dir / name).exists():
            raise SystemExit(
                f"[ACT] missing {name} in {model_dir}. Run `python setup_demos.py ACT` first."
            )

    g = np.load(model_dir / "glue_params.npz")
    D = int(g["D"])
    img_path = Path(args.image) if args.image else model_dir / "sample_image_i16.bin"
    state_path = Path(args.state) if args.state else model_dir / "sample_state.bin"
    for p in (img_path, state_path):
        if not p.exists():
            raise SystemExit(f"[ACT] missing input {p}; pass --image/--state or run setup to fetch the sample frame.")
    image = np.fromfile(img_path, np.int16).reshape(1, 480, 640, 3)
    state = np.fromfile(state_path, np.float32).reshape(1, D)

    logger.info("Loading the three vmfbs on the torq device ...")
    backbone = VMFBInferenceRunner(str(model_dir / "backbone_fused.vmfb"))
    enc4 = VMFBInferenceRunner(str(model_dir / "enc4_dattn.vmfb"))
    dec = VMFBInferenceRunner(str(model_dir / "decoder_ffn.vmfb"))

    t0 = time.time()
    # 1) backbone (NSS): int16 image -> bf16 image features (permute_1)
    permute_1 = backbone.infer([image])[0]
    p1 = _to_f32(permute_1).reshape(300, 1, 512)

    # 2) host glue (numpy): permute_1 + state -> add_50, stack_2 (bf16)
    add_50, stack_2 = host_glue(p1, state, g)

    # 3) encoder(4) + decoder cross-attn (NSS): (add_50, stack_2) -> layer_norm_9
    ln9 = enc4.infer([_to_bf16(add_50), _to_bf16(stack_2)])[0]

    # 4) decoder FFN + action head (NSS): layer_norm_9 -> action
    action_bf16 = dec.infer([_to_bf16(_to_f32(ln9))])[0]
    dt = time.time() - t0

    action = _to_f32(action_bf16).reshape(H, D) * g["a_std"] + g["a_mean"]
    action.astype(np.float32).tofile(model_dir / "action.bin")

    print(f"[ACT] done in {dt * 1e3:.0f} ms "
          f"(backbone {backbone.infer_time_ms:.0f} / enc+xattn {enc4.infer_time_ms:.0f} / "
          f"dec_ffn {dec.infer_time_ms:.0f} ms NSS)")
    print(f"[ACT] action chunk: shape {action.shape}  (wrote {model_dir / 'action.bin'})")
    np.set_printoptions(precision=3, suppress=True)
    print("[ACT] action[0] :", action[0])
    print("[ACT] action[-1]:", action[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ACT policy on the Torq NSS.")
    parser.add_argument(
        "-m", "--model", type=str, default=None, metavar="DIR",
        help="Model directory (default: models/<repo_id> set up by `setup_demos.py ACT`)",
    )
    parser.add_argument(
        "--image", type=str, default=None, metavar="FILE",
        help="int16 NHWC [1,480,640,3] image .bin (default: the bundled sample frame)",
    )
    parser.add_argument(
        "--state", type=str, default=None, metavar="FILE",
        help="float32 [1,D] normalized state .bin (default: the bundled sample state)",
    )
    parser.add_argument(
        "--no-refresh", action="store_true", default=False,
        help="Skip the Hugging Face check for updated models (offline/airgapped runs)",
    )
    add_logging_args(parser)
    main(parser.parse_args())
