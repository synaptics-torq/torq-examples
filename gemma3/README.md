# Gemma 3 Demo

Interactive chat with Gemma 3 270M using Torq VMFB models.

## Setup

From the repo root, run:

```sh
python setup_demos.py gemma3
```

This downloads the default instruct model files to:

```sh
models/Synaptics/gemma-3-270m-it-torq/
```

## Running

Run the demo from the `gemma3` directory:

```sh
cd gemma3
python src/infer.py -m ../models/Synaptics/gemma-3-270m-it-torq/model.vmfb.trim --instruct-model
```

If your downloaded model directory only contains `model.vmfb`, use that file instead.

Type `exit` or `quit` to stop the chat session. While an answer is being generated, press <kbd>Ctrl</kbd> + <kbd>C</kbd> or <kbd>Ctrl</kbd> + <kbd>D</kbd> to interrupt it and return to the prompt.

Run `python src/infer.py -h` to see all available inference options.

## Validation

Gemma 3 includes a validation script for text translation datasets. For example:

```sh
cd gemma3
python src/validate.py -m ../models/Synaptics/gemma-3-270m-it-torq/model.vmfb.trim --instruct-model --max-samples 10
```

Run `python src/validate.py -h` to see all available validation options.
