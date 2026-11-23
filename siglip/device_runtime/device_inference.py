#!/usr/bin/python3
import argparse
import torq.runtime
import ml_dtypes
import numpy as np
import time

from PIL import Image


def load_image(image_path):
    IMAGE_SIZE = 256
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)    
    pixels = np.array(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    return pixels


def patchify(pixels):
    PATCH = 32
    H, W, C = pixels.shape
    # Reshape to [H/P, P, W/P, P, C]
    x = pixels.reshape(H // PATCH, PATCH, W // PATCH, PATCH, C)
    # Transpose to [H/P, W/P, P, P, C]
    x = x.transpose(0, 2, 1, 3, 4)
    # Flatten to [NUM_PATCHES, PATCH * PATCH * C]
    return x.reshape(-1, PATCH * PATCH * C)


def main():
    # Handle command line arguments for --model --input and --output
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, help="Path to the model file")
    parser.add_argument("--res_models", type=str, help="Residual functions")
    parser.add_argument("--input", type=str, default="Doge.jpg", help="Path to the input file")
    parser.add_argument("--output", type=str, default="siglip2_tokens.npy", help="Path to the output npy file")
    args = parser.parse_args()

    runners = [
        torq.runtime.VMFBInferenceRunner("patchify_matmul.vmfb"),
        torq.runtime.VMFBInferenceRunner("pos_emb_add.vmfb"),
    ]

    residual_add_runner = torq.runtime.VMFBInferenceRunner("residual_add.vmfb")

    residual_runners = []
    gelu_runner = torq.runtime.VMFBInferenceRunner("gelu.vmfb")
    for i in range(12):
        runner_block = [
            torq.runtime.VMFBInferenceRunner(f"mha_residual_{i}.vmfb")
        ]
        residual_runners.append(runner_block)
        runner_block = [
            torq.runtime.VMFBInferenceRunner(f"feedforward_residual_a_{i}.vmfb"),
            gelu_runner,
            torq.runtime.VMFBInferenceRunner(f"feedforward_residual_b_{i}.vmfb"),
        ]
        residual_runners.append(runner_block)

    post_layer_norm_runner = torq.runtime.VMFBInferenceRunner(
        "post_layer_norm.vmfb")

    # Load Image
    print(f"Loading image {args.input}...")
    input_data = load_image(args.input)
    input_data = patchify(input_data)
    input_data = input_data.astype('bfloat16')

    start_time = time.perf_counter()
    x = input_data
    # Patchify stem + PosEmbeding
    for i, runner in enumerate(runners):
        x = runner.infer([x])[0]

    # Transformer Tower
    for i, runner_block in enumerate(residual_runners):
        identity = x
        for runner in runner_block:
            x = runner.infer([x])[0]
        x = residual_add_runner.infer([x, identity])[0]

    x = post_layer_norm_runner.infer([x])[0]

    end_time = time.perf_counter()

    print(f"Elapsed time: {end_time - start_time:.6f} seconds")

    np.save(args.output, x.astype(np.float32))
    print(f"Saved tokens to {args.output}")

if __name__ == "__main__":
    main()