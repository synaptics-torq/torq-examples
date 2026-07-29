import numpy as np
from pathlib import Path

TEST_DIRECTORY = Path("test")

HEAD_SHAPES = {
    (1, 1, 20, 20): "cls_scores_s16",
    (1, 1, 10, 10): "cls_scores_s32",
    (1, 4, 20, 20): "bbox_preds_s16",
    (1, 4, 10, 10): "bbox_preds_s32",
    (1, 17, 20, 20): "kpt_vis_s16",
    (1, 17, 10, 10): "kpt_vis_s32",
    (1, 192, 20, 20): "pose_feats_s16",
    (1, 192, 10, 10): "pose_feats_s32",
}

def unpack_heads(outputs, dtype=np.float32):
    """vmfb outputs -> {name: fp32 array}, matched by shape, not position."""
    heads = {}
    for o in outputs:
        a = np.asarray(o).astype(dtype)
        name = HEAD_SHAPES.get(a.shape)
        if name is None:
            raise ValueError(f"unexpected output shape {a.shape}")
        if name in heads:
            raise ValueError(f"duplicate output for {name}")
        heads[name] = a
    missing = set(HEAD_SHAPES.values()) - set(heads)
    if missing:
        raise ValueError(f"missing outputs: {sorted(missing)}")
    return heads


def resolve_input_image(filename):
    """
    Resolve an image filename inside the test directory.

    Both of these are accepted:
        000000000785.jpg
        test/000000000785.jpg
    """
    image_path = Path(filename)

    if image_path.parts and image_path.parts[0] == TEST_DIRECTORY.name:
        image_path = Path(*image_path.parts[1:])

    input_path = TEST_DIRECTORY / image_path

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input image was not found: '{input_path}'"
        )

    return input_path