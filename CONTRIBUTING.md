# Contributing to examples repository

## Types of examples

Each example belongs to one of the following task categories based on inputs and outputs:

- object_detection: extract bounding boxes of objects from an image
- pose_estimation: extract keypoints of poses from an image
- text_generation: generate text based on a given text prompt
- visual_question_answering: generate text answers about a given image
- speech_recognition: transcribe audio to text

It is possible to add new task categories as needed.

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
5. A script `src/validate.py` that allows to validate the example output agains known correct results.
6. An `info.json` file that describes the example, including metadata such as the example name, description, and task category.

The setup demo_script entry point must use the `utils.model_setup.demo_main` function.

The infer script entry point must use the `utils.tasks.[task_category].infer_main` function.

The validate script entry point must use the `utils.tasks.[task_category].validate_main` function.

The info file must be a valid JSON file and should include at least the following fields:

- `name`: the name of the example
- `task_category`: the task category the example belongs to
- `supported_torq_versions`: a list of Torq versions that the example supports

The info file is used by the test framework to run appropriate tests for the example.

## Adding a new task category

To add a new task category:

1. Create a new folder for the task category under `utils/tasks/`.
2. Inside the new folder, create an `__init__.py` file to make it a Python package.
3. Implement the `infer_main` and `validate_main` functions.
4. Add at least one test in the `tests/tasks/[task_category]/` folder for the new task category.

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
     {"name": "tokens_per_second", "description": "Time taken to generate all answer tokens divided by total number of tokens", "unit": "hz"},
     {"name": "cpu_load", "description": "CPU load during demo execution", "unit": "percent"},
     {"name": "npu_load", "description": "NPU load during demo execution", "unit": "percent"}
   ],
   "measurements" : {
     "time_to_first_token": 100, "tokens_per_second": 10, "cpu_load": 0.1, "npu_load": 0.9
    }
}
```

where unit is one of:

- `ns` (nanoseconds)
- `hz` (rates)
- `percent` (percentage)
- `bytes` (memory usage)