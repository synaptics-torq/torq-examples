# TorqExamplesTest

This package is the host-side execution layer for the TorqExamples demo validation flow.

It will push torq-examples and related test materials to DUT-side. Auto tests are run on DUT side. All logs files, pytest result XML and test result HTML will save on host-side

The goal is to drive the actual demo execution on a DUT through the existing board/runner abstraction.

The repository-local pytest plugin is loaded automatically from `conftest.py`.
The executor prepends `TorqExamplesTest` to `sys.path`, so the bundled
`TorqExamplesTest/torq` host board utilities are imported as the
top-level `torq` namespace instead of an unrelated package with the same name
from site-packages. Both supported layouts are detected automatically:
`TorqExamplesTest` beside `torq-examples`, or copied inside the
torq-examples repository.

## Scope

- resolve the latest demo names from `torq-examples/setup_demos.py`
- map each demo to its Python inference entrypoint
- acquire one DUT through Torq board control, retain its lock for the complete pytest session, and reuse it for every stage
- run each demo through the host-side board/runner execution path
- profile every generated model on the DUT over ADB and collect CPU, DRAM, NPU, latency, and memory metrics
- dynamically parse `setup_demos.py` and each demo's `setup_demo.py` to discover selected model aliases and Hugging Face repo IDs, derive DUT model directories, and save them to `logs/setup_model_paths.json`
- run `setup_demos.py` without `--model-version`, allowing torq-examples to use its release `VERSION` for built-in repositories and the latest HEAD revision for custom repositories
- after setup, read each model directory's `.manifest.json`, print its Repo ID, version, and revision, and save the results to `logs/setup_model_versions.json` and the JUnit `model_versions` property; custom repositories without a pinned version are reported as `HEAD (untracked)`
- during prepare, read the actual pushed torq-examples Git commit from the DUT
	and `SYNA_SDK_REVISION` from DUT `/etc/buildinfo`, then save both as JUnit
	properties and `logs/prepare_metadata.json` for combined HTML generation
- parameterize conversion, inference, and profiling as one pytest case per dynamically discovered model repository
- save conversion, inference, profiling, and failure artifacts separately for every model case using a `demo_repo-id` filename prefix, preventing models from overwriting one another's logs or profiling JSON
- collect pass/fail results on a per-demo basis
- avoid adding shell-only assumptions into the execution contract

## Execution model

- `demo_registry.py`: resolves the latest demo inventory and entrypoints
- `demo_executor.py`: performs the real host-side demo execution flow
- `test_demo_execution.py`: exercises the per-model run and result collection
- `conftest.py`: repository-local pytest plugin that cleans artifacts, tees the
	console log, writes reports, and preserves report-generation failure semantics
- `generate_test_summary.py`: merges JUnit results with structured profiling records

## Run and report generation

Run the complete suite from the workspace root:

	python -m pytest TorqExamplesTest

The `pytest.ini` configuration enables stdout streaming and JUnit output for a
bare pytest invocation. Before tests start, the plugin removes artifacts left
by the previous run:
the configured JUnit XML and combined HTML, plus all `.log`, `.json`, `.xml`,
and `.html` files under `torq-examples/logs`. This prevents stale test or
profiling data from being included in the new report.

The plugin preserves pytest's exit code after report generation. If pytest
succeeds but combined-report generation fails, the session is changed to a
failure. Every invocation produces:

- `TEST-TorqExamplesTest.xml`: final pytest JUnit report
- `test_and_profile_summary.html`: combined test and profiling report
- `torq-examples/logs/main.log`: complete suite console output, including
	runner messages plus merged pytest stdout and stderr; output remains visible
	in the Jenkins console in real time

The output locations can be overridden with `--torq-profile-dir` and
`--torq-summary-html`. The JUnit location can be overridden with pytest's
standard `--junitxml` option.

The HTML contains the Torq Examples commit, DUT SDK version,
per-case PASS/FAIL/SKIP results, each model's VERSION and resolved Hugging Face
commit revision, and the seven profiling metrics. Model version data is read
from the JUnit `model_versions` properties with
`torq-examples/logs/setup_model_versions.json` as a fallback. Profiling data
is read first from the JUnit `profile_metrics` properties and then supplemented
from `torq-examples/logs/*_profile.json` when necessary.

To regenerate only the HTML from existing artifacts:

	python TorqExamplesTest/generate_test_summary.py --junit TEST-TorqExamplesTest.xml --profile-dir torq-examples/logs --out test_and_profile_summary.html

## Important boundary

This layer is intentionally not a validation suite for the refactor helper implementation itself. The object under test is the real demo execution flow on the DUT, not the Python utility code used to orchestrate it.
