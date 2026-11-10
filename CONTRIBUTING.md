# Contributing to examples repository

## Types of examples

Each example belongs to one of the following task categories based on inputs and outputs:

- object_detection: extract bounding boxes of objects from an image
- keypoint_detection: extract keypoints of poses from an image
- face_recognition: extract bounding boxes of faces from an image
- text_generation: generate text based on a given text prompt
- image_text_to_text: generate text answers about a given image
- automatic_speech_recognition: transcribe audio to text
- feature_extraction: embed text, images, video and audio into vectors that can be compared and ranked

It is possible to add new task categories as needed. Category names are [Hugging Face task](https://huggingface.co/tasks) ids written with underscores (`text-generation` -> `text_generation`); `face_recognition` has no Hugging Face equivalent.

## Structure of an example

Each example lives in its own folder.

Examples are python scripts that run in its own virtual environment on the target system.

The virtual environment contains the packages listed in the top level `requirements.txt` file and the example's own `requirements.txt` file.

The examples can use the package `utils` in the top level directory that contains common utility functions.

An example folder must contain:

1. A `README.md` file that describes the example.
2. A `requirements.txt` file that lists the dependencies for the example. The dependencies must be pinned to specific versions.
3. A script `setup_demo.py` that downloads all the model files required for the example.
4. A script `src/infer.py` that allows to execute the example
5. A script `src/validate.py` that allows to validate the example output against known correct results.
6. An `info.json` file that describes the example, including metadata such as the example name, description, and task category.

The setup demo_script entry point must use the `utils.model_setup.demo_main` function.

The setup module must expose a uniform entry point `setup(models=None, *, model_version=None, no_update=False, enable_prefill=False)`; the top-level `setup_demos.py` discovers demos from their `info.json` files and calls it. Demos without a batched prefill model accept and ignore `enable_prefill` (they also omit the `--with-batch-prefill` flag, which `demo_main` only adds for `supports_prefill=True`).

The infer script entry point must use the `utils.tasks.[task_category].infer_main` function.

The validate script entry point must use the `utils.tasks.[task_category].validate_main` function.

The info file must be a valid JSON file and should include at least the following fields:

- `name`: the name of the example
- `description`: a one-sentence description of the example
- `task_category`: the task category the example belongs to
- `validation`: the Hugging Face datasets `src/validate.py` scores by default, each `{"dataset", "config", "split", "samples"}` (`config` and `split` for datasets read through the dataset viewer); the task category, or the demo's `validate.py`, must have a loader for each

Demos with a batched prefill model set `"batch_prefill": true`; `setup_demos.py --with-batch-prefill` tracks the extra model for those demos and warns for the rest.

The info file is used by the test framework to run appropriate tests for the example.

## Inference core

The demos are thin scripts over a shared inference core in `utils/`; a demo's `src/runner.py` (or equivalent) composes the layers below, and its `src/infer.py`/`src/validate.py` call the task category's `infer_main`/`validate_main`.

**Runtime session (`utils/runtime.py`).** `RuntimeConfig` is the frozen runtime options of one session, built once from the CLI args (`add_common_inference_args` adds the shared runtime group: `--tda` allocator, `--device-io`, `--runtime-flags`, `-j` threads, plus the logging and `--metrics-out` args). `Model` is the only place a `torq.runtime.VMFBInferenceRunner` is constructed; with device I/O on, `infer` stages the host inputs on the device and returns the raw outputs, and `run` copies them home. `npu_clock()` holds the NPU at its max userspace frequency for the inference block and restores the minimum on exit. The dmabuf allocator is incompatible with host NumPy I/O (a known runtime bug), so `RuntimeConfig` forces device I/O on for `--tda dmabuf`; do not remove that forcing.

**KV cache (`utils/cache.py`).** `KVCache` owns the trailing cache inputs of one or more models that share a layout (a decode body plus its batched prefill body). It zero-initialises every slot on the device from the model's input metadata and exposes them as a mutable `inputs` list the decoder appends after the model's own inputs; `update` folds a model call's outputs back in place, so every attached model sees the very same on-device buffers. For read-only KV cache models the outputs hold only the step's new rows, written at the step position in one command-buffer submission. `fixed` slots hold external state such as encoder-decoder cross-attention KV, and `snapshot`/`restore`/`reset` support the LLM's rolling window.

**Decoding (`utils/decoder.py`).** `Decoder` composes a single-token decode `Model`, its `KVCache`, an optional split LM head and an optional fixed-size batched prefill model, and preallocates every per-step input buffer from the model's input metadata. `step` runs one decode token and returns the host logits (or None, skipping the LM head); `prefill` feeds a prompt through the prefill chunks or single steps; `sample` maps host logits to a token (greedy at temperature 0, top-k then top-p otherwise); `generate` is the one autoregressive loop, with stop, max-tokens and capacity hooks, that the LLM, VLM and STT demos run.

**Chat (`utils/llm.py`).** `LLM` is a `Decoder` plus the conversation state. It loads the model directory (the body VMFB, an optional sibling `lm_head.vmfb` and `transformer_prefill.vmfb`, the embedding and vocab LUTs, `config.json` and `tokenizer.json`), prefills the system prompt once as a warm-up prefix that every conversation reuses, keeps the KV cache shared across turns, and rolls it when the conversation no longer fits (the warm-up plus `cache_keep_n` recent tokens, replayed at their new positions). `ChatTemplate` is plain data — the strings that format one chat turn and open the model's answer, plus the stop tokens — that tells chat formats apart; a model family that differs only in data gets a new template, not a new class. `run`/`run_stream` generate one answer (streamed as decoded text chunks).

**Metrics (`utils/metrics.py`).** `MetricsRun` is a context manager around a demo's inference block: `add(name, value)` records the named measurements (repeated names are averaged) while a sampler collects the CPU load, NPU load and peak memory. At exit it writes the `{"metrics": [...], "measurements": {...}}` JSON shown below to the `--metrics-out` file, even when the block is interrupted.

**Task categories (`utils/tasks/`).** One package per task category (`utils/tasks/<category>/__init__.py`) with the shared `infer_main`/`validate_main` the demo's scripts call: the demo's own flags, default-model resolution and refresh, one NPU-boosted session, the demo's per-input loop, and the category's validation scoring. The image categories share their validation and box-matching plumbing in `utils/tasks/vision.py`.

## Composition rules

- No mixins. Optional behaviour is a constructor argument (`LLM(chat=, template=, cache_keep_n=, ...)`), and stateless logic is a function (`sample`, `generate`, the qwen3 `hide_thinking`), not a class.
- At most one subclass level below a core class, only for model-specific hooks: each demo's runner subclasses its core once (`Gemma3Static(LLM)` only binds a template; `FunctionGemmaStatic(LLM)` adds the function-calling hooks).
- Model families that differ in data get a `ChatTemplate` (the templates in `utils/llm.py`, adjusted with `dataclasses.replace`), not a new class.
- No class inherits from `torq.runtime.VMFBInferenceRunner`; `utils.runtime.Model` wraps it in composition.
- The hot path is flat: the per-step buffers are preallocated from the model's input metadata, and the step loop does no per-step allocation or `isinstance` dispatch.

## Validation data

The repository ships no validation data. `src/validate.py` downloads the rows it scores from public Hugging Face datasets (`utils/datasets.py`), so every dataset stays under its own license and terms, stated on its dataset card:

- A demo's default datasets and sample counts are the `validation` list in its `info.json`; `--dataset`, `--config`, `--split` and `--samples` select others.
- Rows come from the Hugging Face dataset viewer API (`fetch_rows`: exactly the requested rows, media saved as files by data type). Datasets the viewer does not serve are read as raw repo files (`fetch_file`). Both cache under `$MODELS/datasets/`, so repeat runs are offline and the folder can be copied to a board without network access.
- Each task category maps the datasets it supports to loaders (`LOADERS` in `utils/tasks/<category>/`) that turn rows into scoring items; a demo-specific dataset (for example one tied to the demo's output format) is a loader passed by the demo's `validate.py`.
- Prefer the dataset the model card reports. Before adding one, check the existing loaders and the other demos' `info.json`: a dataset your demo needs may already be supported, so reusing it is one `info.json` entry.
- Scoring is against the dataset's own labels; nothing model-generated (goldens) is stored in the repository.

## Adding a new task category

To add a new task category:

1. Create the category module `utils/tasks/<category>/__init__.py` (a new package under `utils/tasks/`).
2. Implement the `infer_main` and `validate_main` functions in it (the shared category plumbing lives in `utils/tasks/__init__.py` and `utils/tasks/vision.py`).
3. Add a `LOADERS` dict mapping the Hugging Face datasets the category supports to loaders that turn their rows into scoring items (see "Validation data").
4. Add at least one test in the `tests/tasks/<category>/` folder.

The tests are run on the host, in an environment where a torq-compiler wheel is installed. They
run compilation of models in an appropriate environment for each example 
(see `requirements_compile.txt`) and setup and execute the examples in a target board
using the `torq.utils.boards` tooling.

Tests have two phases:

1. Setup phase: upload example to the target board and run any necessary setup commands, then 
   download all assets required for the test ( input data, expected results).
2. Execution phase: run the inference for the test on the target board and compare the output against the expected results.

Tests rely on the command line interface provided by the example scripts to execute the setup and validation phases.

The execution phase outputs a PASS/FAIL result based on some criteria (e.g. comparison with expected results) and
performance metrics in the following format:

```json
{
   "metrics":  [
     {"name": "time_to_first_token", "description": "Time taken to generate the first token", "unit": "ns"},
     {"name": "tokens_per_second", "description": "Generated answer tokens divided by the time taken to generate them", "unit": "hz"},
     {"name": "cpu_load", "description": "CPU load during demo execution", "unit": "percent"},
     {"name": "npu_load", "description": "NPU load during demo execution", "unit": "percent"}
   ],
   "measurements" : {
     "time_to_first_token": 100, "tokens_per_second": 10, "cpu_load": 10, "npu_load": 90
    }
}
```

where unit is one of:

- `ns` (nanoseconds)
- `hz` (rates)
- `percent` (percentage; values are 0-100, so 90 means 90%)
- `bytes` (memory usage)