"""
NumPy reimplementation of the RTMO-320 ONNX postprocessing subgraph.

Replaces nodes 333..640 of model_320.onnx. Consumes the 8 raw head Conv
outputs and produces `dets` [B,N,5] and `keypoints` [B,N,17,3].

Weights come from postprocess_weights.npz (constants lifted out of the graph).
"""

import numpy as np
from pathlib import Path

NUM_KEYPOINTS = 17
HIDDEN = 128          # DCC per-keypoint feature dim
GAU_EXPAND = 640      # 256 (u) + 256 (v) + 128 (base for q/k)
X_BINS = 192          # SimCC x bins
Y_BINS = 256          # SimCC y bins

# From the graph constants
NMS_TOP_K = 200
SCORE_THR = 0.10000000149011612
IOU_THR = 0.6499999761581421
PRE_TOPK = 2000       # pre-NMS cap (node 74/75)
POST_TOPK = 50        # post-NMS cap (node 163/164)
BBOX_EXPAND = 1.25    # bbox->center/scale widening (node 213)
GAU_DIV = 11.313708305358887   # sqrt(128)
GAU_LN_EPS = 9.999999747378752e-06
GAU_LN_SCALE = 0.0883883461356163   # 1/sqrt(128)
SOFTMAX_EPS = 9.99999993922529e-09
LOGIT_CLIP = 50000.0

POSTPROCESS_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = POSTPROCESS_DIR / "postprocess_weights.npz"

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _flatten_level(t, channels):
    """[B,C,H,W] -> [B,H*W,C]  (ONNX Transpose(0,2,3,1) + Reshape)."""
    b = t.shape[0]
    return t.transpose(0, 2, 3, 1).reshape(b, -1, channels)


def _softmax_last(x):
    """Matches the graph exactly: clip, subtract max, exp, divide by (sum+eps)."""
    x = np.clip(x, -LOGIT_CLIP, LOGIT_CLIP)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + SOFTMAX_EPS)


def _nms_per_class(boxes, scores, iou_thr, score_thr, max_out):
    """
    Mirrors ONNX NonMaxSuppression (center_point_box=0, corner format).
    boxes:  [B, N, 4] as (x1, y1, x2, y2)
    scores: [B, C, N]
    Returns [K, 3] int64 rows of (batch, class, box_index), like the ONNX op.
    """
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
                xx1 = np.maximum(x1[i], x1[rest])
                yy1 = np.maximum(y1[i], y1[rest])
                xx2 = np.minimum(x2[i], x2[rest])
                yy2 = np.minimum(y2[i], y2[rest])
                inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
                iou = inter / (area[i] + area[rest] - inter)
                order = rest[iou <= iou_thr]
            out.extend([(b, c, int(i)) for i in keep])
    if not out:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(out, dtype=np.int64)


def postprocess(cls_20, cls_10, bbox_20, bbox_10, pose_20, pose_10,
                kptvis_20, kptvis_10, w):
    """
    Args match the extracted ONNX subgraph inputs:
      cls_20    [B,  1, 20,20]   cls_10    [B,  1, 10,10]
      bbox_20   [B,  4, 20,20]   bbox_10   [B,  4, 10,10]
      pose_20   [B,192, 20,20]   pose_10   [B,192, 10,10]
      kptvis_20 [B, 17, 20,20]   kptvis_10 [B, 17, 10,10]
      w: dict-like of weights (np.load of postprocess_weights.npz)

    Returns (dets [B,N,5], keypoints [B,N,17,3]).
    """
    B = cls_20.shape[0]

    # ---- 1. Flatten + concat both FPN levels (nodes 0..47) ----
    scores_all = _sigmoid(np.concatenate(
        [_flatten_level(cls_20, 1), _flatten_level(cls_10, 1)], axis=1))          # [B,500,1]
    bbox_all = np.concatenate(
        [_flatten_level(bbox_20, 4), _flatten_level(bbox_10, 4)], axis=1)         # [B,500,4]
    pose_all = np.concatenate(
        [_flatten_level(pose_20, 192), _flatten_level(pose_10, 192)], axis=1)     # [B,500,192]
    kptvis_all = _sigmoid(np.concatenate(
        [_flatten_level(kptvis_20, 17), _flatten_level(kptvis_10, 17)], axis=1))  # [B,500,17]

    # ---- 2. Decode boxes: ctr = pred*stride + grid, wh = exp(pred)*stride (48..68) ----
    ctr = bbox_all[..., :2] * w["grid_stride"] + w["grid_offset"]
    wh = np.exp(bbox_all[..., 2:]) * w["grid_stride"]
    half = wh / 2.0
    boxes_all = np.concatenate([ctr - half, ctr + half], axis=-1)                 # [B,500,4] xyxy

    # ---- 3. Pre-NMS TopK on max class score (71..99) ----
    max_scores = scores_all.max(axis=-1)                                          # [B,500]
    k = min(PRE_TOPK, max_scores.shape[1])
    topk_inds = np.argsort(-max_scores, axis=1, kind="stable")[:, :k]             # [B,k]

    bt = np.take_along_axis(boxes_all, topk_inds[..., None], axis=1)              # [B,k,4]
    st = np.take_along_axis(scores_all, topk_inds[..., None], axis=1)             # [B,k,1]
    scores_t = st.transpose(0, 2, 1)                                              # [B,1,k]

    # ---- 4. NMS (100) ----
    sel = _nms_per_class(bt, scores_t, IOU_THR, SCORE_THR, NMS_TOP_K)

    # ---- 5. Gather selections, pad to fixed width, post-NMS TopK (101..174) ----
    # Graph builds [B, max_dets, 5] padded with 0 (scores) / -1 (indices).
    dets_list, idx_list = [], []
    for b in range(B):
        rows = sel[sel[:, 0] == b]
        bi = rows[:, 2]
        d = np.concatenate([bt[b, bi], scores_t[b, rows[:, 1], bi][:, None]], axis=1)
        # graph appends one all-zero row (ConstantOfShape at 141) before the post-TopK
        d = np.concatenate([d, np.zeros((1, 5), np.float32)], axis=0)
        oi = np.concatenate([topk_inds[b, bi], np.array([-1], np.int64)])
        n = min(POST_TOPK, d.shape[0])
        order = np.argsort(-d[:, 4], kind="stable")[:n]
        dets_list.append(d[order])
        idx_list.append(oi[order])
    N = max(d.shape[0] for d in dets_list)
    keep_idx = np.full((B, N), -1, np.int64)
    for b in range(B):
        keep_idx[b, :idx_list[b].shape[0]] = idx_list[b]

    # ---- 6. Re-gather everything from the full 500 using keep_idx (177..208) ----
    # NOTE: the graph gathers with the raw index, so the -1 pad wraps to the LAST
    # anchor (499) rather than producing a zero row. Reproduced here for bit-exactness.
    wrap = keep_idx % boxes_all.shape[1]
    dets = np.concatenate(
        [np.take_along_axis(boxes_all, wrap[..., None], axis=1),
         np.take_along_axis(scores_all, wrap[..., None], axis=1)], axis=-1)       # [B,N,5]
    pose_sel = np.take_along_axis(pose_all, wrap[..., None], axis=1)              # [B,N,192]
    vis_sel = np.take_along_axis(kptvis_all, wrap[..., None], axis=1)             # [B,N,17]
    prior_sel = w["priors"][wrap]                                                 # [B,N,2]

    # ---- 7. bbox -> center/scale, scale widened by 1.25 (209..216) ----
    x1y1, x2y2 = dets[..., 0:2], dets[..., 2:4]
    center = (x2y2 + x1y1) * 0.5
    scale = (x2y2 - x1y1) * BBOX_EXPAND
    # graph works in prior-relative coords, then adds the prior back at the end
    origin = center - prior_sel

    # ---- 8. pose feat -> per-keypoint tokens (217..225) ----
    kpt_feat = pose_sel @ w["pose_to_kpts_w"] + w["pose_to_kpts_b"]               # [B,N,2176]
    kpt_feat = kpt_feat.reshape(B, N, NUM_KEYPOINTS, HIDDEN)

    # ---- 9. GAU block: ScaleNorm -> gated attention -> residual (226..251) ----
    # NOTE: Abs feeds only the norm; the divide and the residual use the raw tensor.
    norm = np.sqrt((np.abs(kpt_feat) ** 2).sum(axis=-1, keepdims=True)) * GAU_LN_SCALE
    xn = (kpt_feat / np.clip(norm, GAU_LN_EPS, None)) * w["gau_ln_g"]

    uvg = xn @ w["gau_uvg_w"]
    uvg = uvg * _sigmoid(uvg)                                                     # SiLU
    u, v, base = np.split(uvg, [256, 512], axis=-1)

    q = base * w["q_scale"] + w["q_bias"]
    k_ = base * w["k_scale"] + w["k_bias"]
    attn = np.einsum("bnid,bnjd->bnij", q, k_) / GAU_DIV
    attn = np.maximum(attn, 0.0) ** 2                                            # ReLU squared
    out = u * (attn @ v)
    feat = kpt_feat * w["res_scale"] + out @ w["gau_o_w"]                        # [B,N,17,128]

    # ---- 10. Build SimCC x/y coordinate bases with RoPE (252..277) ----
    x_coord = w["x_lin"] * scale[..., 0:1] + origin[..., 0:1]                     # [B,N,192]
    y_coord = w["y_lin"] * scale[..., 1:2] + origin[..., 1:2]                     # [B,N,256]

    def _rope_fc(coord, fc_w, fc_b):
        t = coord[..., None] / w["rope_div"]
        emb = np.concatenate([np.cos(t), np.sin(t)], axis=-1)
        return (emb @ fc_w + fc_b).transpose(0, 1, 3, 2)

    x_basis = _rope_fc(x_coord, w["x_fc_w"], w["x_fc_b"])                         # [B,N,128,192]
    y_basis = _rope_fc(y_coord, w["y_fc_w"], w["y_fc_b"])                         # [B,N,128,256]

    # ---- 11. SimCC heatmaps -> softmax expectation (278..307) ----
    x_hms = feat @ x_basis                                                        # [B,N,17,192]
    y_hms = feat @ y_basis                                                        # [B,N,17,256]

    px = (_softmax_last(x_hms) * x_coord[:, :, None, :]).sum(-1) + prior_sel[..., 0:1]
    py = (_softmax_last(y_hms) * y_coord[:, :, None, :]).sum(-1) + prior_sel[..., 1:2]

    keypoints = np.stack([px, py, vis_sel], axis=-1)                              # [B,N,17,3]
    return dets, keypoints


def model_postprocess(heads, weights_path=DEFAULT_WEIGHTS_PATH):
    """Run RTMO postprocessing from the named inference head outputs.

    Args:
        heads: Dictionary returned by ``unpack_heads``.
        weights_path: Path to the extracted postprocessing weights.

    Returns:
        Tuple ``(dets, keypoints)`` where:
          * dets has shape [B, N, 5]
          * keypoints has shape [B, N, 17, 3]
    """
    required = (
        "cls_scores_s16",
        "cls_scores_s32",
        "bbox_preds_s16",
        "bbox_preds_s32",
        "pose_feats_s16",
        "pose_feats_s32",
        "kpt_vis_s16",
        "kpt_vis_s32",
    )

    missing = [name for name in required if name not in heads]
    if missing:
        raise KeyError(
            "Missing RTMO output tensor(s): " + ", ".join(missing)
        )

    # The NumPy postprocessing calculations are performed in float32.
    tensors = {
        name: np.asarray(heads[name], dtype=np.float32)
        for name in required
    }

    with np.load(weights_path) as weights:
        return postprocess(
            cls_20=tensors["cls_scores_s16"],
            cls_10=tensors["cls_scores_s32"],
            bbox_20=tensors["bbox_preds_s16"],
            bbox_10=tensors["bbox_preds_s32"],
            pose_20=tensors["pose_feats_s16"],
            pose_10=tensors["pose_feats_s32"],
            kptvis_20=tensors["kpt_vis_s16"],
            kptvis_10=tensors["kpt_vis_s32"],
            w=weights,
        )
