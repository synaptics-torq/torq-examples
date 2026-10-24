# FunctionGemma Demo

Interactive function calling with FunctionGemma 270M Torq VMFB exports. Host regression tests and board comparisons cover BF16, INT4 RTN, and dynamic INT8 exports. BF16 with single-token prefill is the recommended public default for dependability; see the measured tradeoffs under [Precision comparison](#precision-comparison).

## Setup and running

Install the repository dependencies as described in the [root README](../README.md), then install this demo's dependencies:

```sh
pip install -r functiongemma/requirements.txt
```

From the repository root on the board:

```sh
python functiongemma/src/infer.py -m /path/to/compiled/transformer.vmfb --no-refresh --no-batch-prefill-model
```

`-m`/`--model` defaults to the model downloaded during setup. Pass `-m` to use a different (e.g. a custom fine-tuned HF repo) model instead.

> [!NOTE]
> The demo defaults to the DMA/dmabuf allocator with device I/O enabled. Use `--tda cpu` to run with the CPU allocator, or `--no-device-io` to pass user inputs as NumPy arrays.

Type `exit` or `quit` to stop the chat session. While an answer is being generated, press <kbd>Ctrl</kbd> + <kbd>C</kbd> or <kbd>Ctrl</kbd> + <kbd>D</kbd> to interrupt it and return to the prompt.

When the KV cache fills, the sliding window keeps the system prompt and re-prefills up to `--kv-cache-window N` recent tokens at their new positions (default: 2). It retains fewer history tokens when needed to fit the next prompt or generation step. Re-prefilling adds work at rollover, but keeps the cached keys' position encodings correct. With `--no-kv-cache-window`, generation stops at capacity and a later prompt starts a fresh conversation when it no longer fits.

Run `python src/infer.py -h` to see all available inference options.

## Behavior

The runner warms the KV cache with BOS and the developer instruction and tool declarations. It preserves special tokens when decoding function calls, executes one complete call at a time from an explicit function registry, and appends the result without a new chat turn header. The pending `<start_function_response>` token is replayed before the response. A turn can make at most five sequential calls.

The built-in `get_temperature(city)` uses an offline sample table, not live weather. Try `What is the temperature in London?`; a successful call is `get_temperature(city='London')`, returning `18`. Unknown functions and execution failures become tool error responses. The demo does not execute arbitrary code. Add tools to `DEFAULT_TOOLS` and `FUNCTIONS` in `src/function_tools.py`; the declaration renderer supports the demo's object parameters with scalar properties.

`--max-inp-len` includes the developer prefix and the current prompt. Oversized prompts are rejected rather than truncated. Sliding cache windows are disabled because retaining a few trailing tokens loses the function-call context. If a new user prompt cannot fit after the conversation history, the runner starts from its developer prefix. Tool responses that cannot fit, interrupted turns, malformed calls (including calls that omit the required response-start stop), multiple simultaneous calls, and exchanges that exhaust their generation budget discard the conversation before the next user prompt.

> [!NOTE]
> Google documents [single-turn and parallel calls as the model's trained workflows](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices); multi-turn state tracking and dependent chains require additional training or application state management. This demo currently accepts one call at a time and rejects simultaneous calls.
