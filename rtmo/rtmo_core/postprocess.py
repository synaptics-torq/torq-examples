"""NumPy reimplementation of the RTMO-320 ONNX post-processing subgraph.

Consumes the eight raw head Conv outputs and produces ``dets [B,N,5]`` +
``keypoints [B,N,17,3]``, bit-exact vs the graph. Lifted graph constants live
in ``postprocess_weights.npz`` next to this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NUM_KEYPOINTS = 17
HIDDEN = 128          # DCC per-keypoint feature dim
# Constants lifted from the graph:
NMS_TOP_K = 200
SCORE_THR = 0.10000000149011612
IOU_THR = 0.6499999761581421
PRE_TOPK = 2000
POST_TOPK = 50
BBOX_EXPAND = 1.25    # bbox -> center/scale widening
GAU_DIV = 11.313708305358887        # sqrt(128)
GAU_LN_EPS = 9.999999747378752e-06
GAU_LN_SCALE = 0.0883883461356163   # 1/sqrt(128)
SOFTMAX_EPS = 9.99999993922529e-09
LOGIT_CLIP = 50000.0

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "postprocess_weights.npz"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _flatten_level(t, channels):
    """[B,C,H,W] -> [B,H*W,C]."""
    return t.transpose(0, 2, 3, 1).reshape(t.shape[0], -1, channels)


def _softmax_last(x):
    """Graph-exact: clip, subtract max, exp, divide by (sum+eps)."""
    x = np.clip(x, -LOGIT_CLIP, LOGIT_CLIP)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + SOFTMAX_EPS)


def _nms_per_class(boxes, scores, iou_thr, score_thr, max_out):
    """Mirror ONNX NonMaxSuppression: boxes [B,N,4] xyxy, scores [B,C,N] -> [K,3] (batch, class, box)."""
    out = []
    B, C, N = scores.shape
    for b in range(B):
        for c in range(C):
            s = scores[b, c]
            cand = np.where(s > score_thr)[0]
            if cand.size == 0:
                continue
            order = cand[np.argsort(-s[cand], kind="stable")]
            bx = boxes[b]
            x1, y1 = np.minimum(bx[:, 0], bx[:, 2]), np.minimum(bx[:, 1], bx[:, 3])
            x2, y2 = np.maximum(bx[:, 0], bx[:, 2]), np.maximum(bx[:, 1], bx[:, 3])
            area = (x2 - x1) * (y2 - y1)
            keep = []
            while order.size > 0 and len(keep) < max_out:
                i = order[0]
                keep.append(i)
                rest = order[1:]
                if rest.size == 0:
                    break
                inter = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None) * np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
                order = rest[inter / (area[i] + area[rest] - inter) <= iou_thr]
            out.extend([(b, c, int(i)) for i in keep])
    return np.array(out, dtype=np.int64) if out else np.zeros((0, 3), dtype=np.int64)


def postprocess(cls_20, cls_10, bbox_20, bbox_10, pose_20, pose_10, kptvis_20, kptvis_10, w):
    """Decode the eight head outputs to ``(dets [B,N,5], keypoints [B,N,17,3])``; ``w`` = loaded weights npz."""
    B = cls_20.shape[0]

    # 1. Flatten + concat both FPN levels
    scores_all = _sigmoid(np.concatenate([_flatten_level(cls_20, 1), _flatten_level(cls_10, 1)], axis=1))
    bbox_all = np.concatenate([_flatten_level(bbox_20, 4), _flatten_level(bbox_10, 4)], axis=1)
    pose_all = np.concatenate([_flatten_level(pose_20, 192), _flatten_level(pose_10, 192)], axis=1)
    kptvis_all = _sigmoid(np.concatenate([_flatten_level(kptvis_20, 17), _flatten_level(kptvis_10, 17)], axis=1))

    # 2. Decode boxes: ctr = pred*stride + grid, wh = exp(pred)*stride
    ctr = bbox_all[..., :2] * w["grid_stride"] + w["grid_offset"]
    half = np.exp(bbox_all[..., 2:]) * w["grid_stride"] / 2.0
    boxes_all = np.concatenate([ctr - half, ctr + half], axis=-1)

    # 3. Pre-NMS TopK on max class score
    max_scores = scores_all.max(axis=-1)
    topk_inds = np.argsort(-max_scores, axis=1, kind="stable")[:, :min(PRE_TOPK, max_scores.shape[1])]
    bt = np.take_along_axis(boxes_all, topk_inds[..., None], axis=1)
    scores_t = np.take_along_axis(scores_all, topk_inds[..., None], axis=1).transpose(0, 2, 1)

    # 4. NMS
    sel = _nms_per_class(bt, scores_t, IOU_THR, SCORE_THR, NMS_TOP_K)

    # 5. Gather selections, pad (graph appends one zero row), post-NMS TopK
    dets_list, idx_list = [], []
    for b in range(B):
        rows = sel[sel[:, 0] == b]
        bi = rows[:, 2]
        d = np.concatenate([np.concatenate([bt[b, bi], scores_t[b, rows[:, 1], bi][:, None]], axis=1), np.zeros((1, 5), np.float32)], axis=0)
        oi = np.concatenate([topk_inds[b, bi], np.array([-1], np.int64)])
        order = np.argsort(-d[:, 4], kind="stable")[:min(POST_TOPK, d.shape[0])]
        dets_list.append(d[order])
        idx_list.append(oi[order])
    N = max(d.shape[0] for d in dets_list)
    keep_idx = np.full((B, N), -1, np.int64)
    for b in range(B):
        keep_idx[b, :idx_list[b].shape[0]] = idx_list[b]

    # 6. Re-gather from the full anchor set. Graph gathers with the raw index, so
    # the -1 pad wraps to the last anchor — reproduced for bit-exactness.
    wrap = keep_idx % boxes_all.shape[1]
    dets = np.concatenate([np.take_along_axis(boxes_all, wrap[..., None], axis=1), np.take_along_axis(scores_all, wrap[..., None], axis=1)], axis=-1)
    pose_sel = np.take_along_axis(pose_all, wrap[..., None], axis=1)
    vis_sel = np.take_along_axis(kptvis_all, wrap[..., None], axis=1)
    prior_sel = w["priors"][wrap]

    # 7. bbox -> center/scale (widened 1.25)
    x1y1, x2y2 = dets[..., 0:2], dets[..., 2:4]
    scale = (x2y2 - x1y1) * BBOX_EXPAND
    origin = (x2y2 + x1y1) * 0.5 - prior_sel

    # 8. pose feat -> per-keypoint tokens
    kpt_feat = (pose_sel @ w["pose_to_kpts_w"] + w["pose_to_kpts_b"]).reshape(B, N, NUM_KEYPOINTS, HIDDEN)

    # 9. GAU: ScaleNorm -> SiLU(uvg) -> ReLU^2 attention -> residual
    norm = np.sqrt((np.abs(kpt_feat) ** 2).sum(axis=-1, keepdims=True)) * GAU_LN_SCALE
    xn = (kpt_feat / np.clip(norm, GAU_LN_EPS, None)) * w["gau_ln_g"]
    uvg = xn @ w["gau_uvg_w"]
    uvg = uvg * _sigmoid(uvg)
    u, v, base = np.split(uvg, [256, 512], axis=-1)
    attn = np.maximum(np.einsum("bnid,bnjd->bnij", base * w["q_scale"] + w["q_bias"], base * w["k_scale"] + w["k_bias"]) / GAU_DIV, 0.0) ** 2
    feat = kpt_feat * w["res_scale"] + (u * (attn @ v)) @ w["gau_o_w"]

    # 10. SimCC x/y coordinate bases with RoPE
    x_coord = w["x_lin"] * scale[..., 0:1] + origin[..., 0:1]
    y_coord = w["y_lin"] * scale[..., 1:2] + origin[..., 1:2]

    def _rope_fc(coord, fc_w, fc_b):
        t = coord[..., None] / w["rope_div"]
        return (np.concatenate([np.cos(t), np.sin(t)], axis=-1) @ fc_w + fc_b).transpose(0, 1, 3, 2)

    x_basis = _rope_fc(x_coord, w["x_fc_w"], w["x_fc_b"])
    y_basis = _rope_fc(y_coord, w["y_fc_w"], w["y_fc_b"])

    # 11. SimCC heatmaps -> softmax expectation
    px = (_softmax_last(feat @ x_basis) * x_coord[:, :, None, :]).sum(-1) + prior_sel[..., 0:1]
    py = (_softmax_last(feat @ y_basis) * y_coord[:, :, None, :]).sum(-1) + prior_sel[..., 1:2]
    return dets, np.stack([px, py, vis_sel], axis=-1)


def model_postprocess(heads, weights_path=DEFAULT_WEIGHTS_PATH):
    """Decode from a ``{name: array}`` head dict -> ``(dets, keypoints)``."""
    required = ("cls_scores_s16", "cls_scores_s32", "bbox_preds_s16", "bbox_preds_s32", "pose_feats_s16", "pose_feats_s32", "kpt_vis_s16", "kpt_vis_s32")
    missing = [name for name in required if name not in heads]
    if missing:
        raise KeyError("Missing RTMO output tensor(s): " + ", ".join(missing))
    t = {name: np.asarray(heads[name], dtype=np.float32) for name in required}
    with np.load(weights_path) as w:
        return postprocess(t["cls_scores_s16"], t["cls_scores_s32"], t["bbox_preds_s16"], t["bbox_preds_s32"], t["pose_feats_s16"], t["pose_feats_s32"], t["kpt_vis_s16"], t["kpt_vis_s32"], w)
