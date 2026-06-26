# ACT Policy Demo

Run an ACT (Action Chunking Transformer) robot policy on the Torq NSS. From one camera frame +
robot state, the demo produces a 100-step action chunk, end to end on the board.

## Setup

From the repo root, run:

```sh
python setup_demos.py ACT
```

This downloads the model files to:

```sh
models/Synaptics/ACT/
```

The model is optimized as the three separate NSS modules (`backbone_fused.vmfb`, `enc4_dattn.vmfb`,
`decoder_ffn.vmfb`), and includes `glue_params.npz` (the pre-encoder host-glue constants + action
denorm stats) and a bundled sample input frame.

## Running

Note: we highly recommend configuring the chip for best performance as below.
```sh
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo performance > "$p/scaling_governor"
done

# verify
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo -n "$p: "; cat "$p/scaling_governor" "$p/scaling_cur_freq"
done

devmem 0xf7e104b0 32 0x216
```

Run the demo from the `ACT` directory:

```sh
cd ACT
python src/run.py                                            # uses the bundled sample frame
python src/run.py -m ../models/Synaptics/ACT      # explicit model dir
python src/run.py --image img.bin --state state.bin          # your own pre-normalized inputs
```

It prints the denormalized 100-step action chunk + per-stage NSS timing and writes `action.bin`
(float32 `[100,D]`) into the model directory.

Run `python src/run.py -h` to see all options (e.g. `--no-refresh` for offline/airgapped runs).

## What it does

```
backbone_fused.vmfb   int16 image [1,480,640,3] -> bf16 image features         NSS
[numpy host glue]     state proj D->512 + token concat [latent,state,300 img] + positional embedding
enc4_dattn.vmfb       4 encoder layers + decoder cross-attention               NSS
decoder_ffn.vmfb      decoder FFN + action head -> action                      NSS
denormalize           action * a_std + a_mean -> physical joint targets
```

The pre-encoder host glue (state input-projection `D->512`, latent+state+300-image token concat,
additive sin/cos positional embedding) is reproduced in pure numpy and was verified bit-for-bit
against the original onnx subgraph; its constants live in `glue_params.npz` (`Ws, bs, latent, pos`,
plus `a_mean, a_std, D`). The board needs only `torq-runtime`, `numpy`, and `ml_dtypes` (all in the
torq-examples venv) -- no onnxruntime.

> [!NOTE]
> Inputs are raw `.bin` (C-order): image = int16 NHWC `[1,480,640,3]`
> (`round(normalized_image / stem_in_scale)`); state = float32 `[1,D]` (`(state - mean) / std`).
> bf16 tensors between modules are handled with `ml_dtypes.bfloat16`. Inference runs through the
> venv's `torq.runtime.VMFBInferenceRunner` (device `torq`; `--torq_hw_type=astra_machina` is set
> automatically).

Validated on an AstraCORAL-2619 board: ~390 ms on the NSS (backbone 141 / enc+xattn 221 /
dec_ffn 11 ms), reproducing a recorded demonstration to ~0.14 % of joint range.
