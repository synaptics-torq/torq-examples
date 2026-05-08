# SigLIP image tokenizer for SL2610

## Prereqs

Generate models:
```
pushd device_model
./export_vision_params.py
./export_siglip_graphs.py
popd
```

Compile models:
```
torq-compile \
    --mlir-disable-threading \
    patchify_matmul.stablehlo.mlir \
    -o patchify_matmul.stablehlo.vmfb;
torq-compile \
    --mlir-disable-threading \
    pos_emb_add.stablehlo.mlir \
    -o pos_emb_add.stablehlo.vmfb;
torq-compile \
    --mlir-disable-threading \
    residual_add.stablehlo.mlir \
    -o residual_add.stablehlo.vmfb;
torq-compile \
    --mlir-disable-threading \
    gelu.stablehlo.mlir \
    -o gelu.stablehlo.vmfb;
for i in {0..11}; do
    echo "Block $i"
    torq-compile \
        --mlir-disable-threading \
        mha_residual_$i.stablehlo.mlir \
        -o mha_residual_$i.vmfb;

    torq-compile \
        --mlir-disable-threading \
        feedforward_residual_a_$i.stablehlo.mlir \
        -o feedforward_residual_a_$i.vmfb;

    torq-compile \
        --mlir-disable-threading \
        feedforward_residual_b_$i.stablehlo.mlir \
        -o feedforward_residual_b_$i.vmfb;
done
torq-compile \
    --mlir-disable-threading \
    post_layer_norm.stablehlo.mlir \
    -o post_layer_norm.vmfb
```

Push models to device:
```
adb push patchify_matmul.stablehlo.vmfb /home/root
adb push pos_emb_add.stablehlo.vmfb /home/root
adb push gelu.stablehlo.vmfb /home/root
adb push residual_add.stablehlo.vmfb /home/root
adb push mha_residual_*.vmfb /home/root
adb push feedforward_residual_a_*.vmfb /home/root
adb push feedforward_residual_b_*.vmfb /home/root
adb push post_layer_norm.vmfb /home/root
```

## Running a model

On device, run the inference script:

```
./inference.py --input YOUR_IMAGE.jpg --output YOUR_IMAGE.npy
```

Bring the generated tokens back to your host and run

```
./gemma_siglip2_inference.py --mode caption_tokens --token_path YOUR_IMAGE.npy
```