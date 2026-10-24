# Adding FunctionGemma tools

The demo starts with one offline tool, `get_temperature(city)`. You can register other Python functions without changing the runner or retraining the default model. Registration makes a function available to the model but does not guarantee that the model will select it or supply correct arguments. Test your actual requests with each checkpoint and compiled precision you intend to use.

This guide covers the default model and fine-tuned FunctionGemma checkpoints.

## Register a tool

Each tool needs a declaration in `DEFAULT_TOOLS` and a callable under the same name in `FUNCTIONS`, both in [src/function_tools.py](src/function_tools.py). The declaration tells the model what to generate. The registry controls what the host can execute. Python annotations alone do not declare a tool.

For example, add the following after the existing `FUNCTIONS` definition to keep the temperature tool and add an offline country lookup:

```python
def get_country(city: str) -> dict[str, str]:
    if not isinstance(city, str):
        raise ValueError("city must be a string")
    countries = {"london": "United Kingdom", "paris": "France", "tokyo": "Japan"}
    country = countries.get(city.strip().casefold())
    if country is None:
        raise ValueError(f"unknown city '{city}'")
    return {"country": country}


DEFAULT_TOOLS.append({
    "type": "function",
    "function": {
        "name": "get_country",
        "description": "Get the country containing a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name."},
            },
            "required": ["city"],
        },
    },
})
FUNCTIONS["get_country"] = get_country
```

Good function calling practices:
* Use short, distinct function names and descriptions.
* Keep property names identical to the callable's keyword arguments, and list mandatory arguments in `required`.
* The declaration renderer supports object parameters with scalar properties (`string`, `number`, `integer`, `boolean`) and does not render nested property schemas, array item schemas, enums, or defaults. Keep tools within that subset unless you also extend the renderer and its tests. The parser's ability to read nested values does not imply support for declaring their schemas.
* Validate arguments inside the callable. The dispatcher catches missing or unexpected keyword arguments, but it does not enforce schema types or Python annotations. Keep execution in the explicit registry and validate any paths, destinations, or actions a future tool accepts.
* Return small values such as strings, numbers, booleans, lists, or dictionaries with simple identifier keys. 

> [!TIP]
> Expected failures can raise `ValueError`; the CLI sends execution exceptions back to the model as `{"error": "..."}`. Avoid putting secrets into exception messages. 

The CLI loads `DEFAULT_TOOLS` when constructing its runner; restart it after edits. Python callers can instead pass `tools=[...]` to `FunctionGemmaStatic`, but the standard CLI still executes calls through `FUNCTIONS`. A custom declaration list must have matching registered implementations.


## Run with the default model

Install the demo requirements and set up the model as described in the [README](README.md). Copy your edited source to the board's checkout, then run from the repository root on the board:

```sh
.venv/bin/python functiongemma/src/infer.py --no-refresh
```

This uses the already installed default export, with greedy decoding and single-token prefill. `--no-refresh` avoids downloading or refreshing files during the check.

Try each request in a fresh CLI session first:

| Request | Expected call and host result |
| --- | --- |
| What country is Paris in? | `get_country(city='Paris')` -> `{'country': 'France'}` |
| What is the temperature in London? | `get_temperature(city='London')` -> `18` |
| What country is Atlantis in? | `get_country(city='Atlantis')` -> an unknown-city tool error |

Look for the CLI's `[function]` and `[result]` lines since a plausible text answer alone does not prove the tool ran. Note that the calls above are acceptance targets and not guaranteed model outputs. Also try paraphrases, missing arguments, requests needing no tool, and a second user turn. 

Tool declarations consume the compiled model's context along with user prompts, history, generated calls, and tool results. Adding tools can exhaust a small export's capacity. Shorten declarations, replace unused tools in both registries, or use an export compiled for a larger context. Raising `--max-inp-len` or `--max-seq-len` cannot enlarge the compiled KV cache. Oversized prompts are rejected, and partial-history sliding windows are disabled for this demo.

## Run with a fine-tuned checkpoint

> [!WARNING]
> If a checkpoint uses a different chat or tool protocol, it needs runner changes and protocol tests before it can use this demo.

Export and compile the fine-tuned FunctionGemma checkpoint for the target board using the Gemma export layout accepted by this repository. For an adapter checkpoint, produce a complete export with the adapter applied. Keep all artifacts from that export together in a separate directory; do not mix the default model's head, embeddings, tokenizer, or vocabulary mapping into it.

A typical split export looks like this:

```text
my-functiongemma/
  transformer.vmfb
  lm_head.vmfb             # or lm_head.vmfb.trim
  token_embeddings.npy
  tokenizer.json
  config.json
  token_id_lut.npy         # required for compact/trimmed logits
```

The setup helpers also recognize a combined `model.vmfb` or `model.vmfb.trim` instead of the split body/head. Point `-m` at the body or combined model file. Sibling LM heads are discovered automatically; use `--lm-head /path/to/matching/lm_head.vmfb` if discovery is ambiguous. The runtime resolves `config.json`, `tokenizer.json`, embeddings, and any vocabulary lookup beside the model passed to `-m`.

Preserve the FunctionGemma special tokens, including `<start_function_call>`, `<end_function_call>`, `<start_function_response>`, `<end_function_response>`, and `<escape>`. The exported `config.json` must include the tokenizer's `<start_function_response>` ID in `eos_token_id`, alongside the checkpoint's other stop IDs. The runtime reads stop IDs from this file, not `generation_config.json`; do not copy numeric IDs from another tokenizer. The runner checks the function markers and response-start stop at startup.

Run the fine-tuned export on the board:

```sh
.venv/bin/python functiongemma/src/infer.py -m /path/to/my-functiongemma/transformer.vmfb --no-refresh
```

## Keep the existing execution contract

The host executes one complete, registered call at a time, only after generation stops on `<start_function_response>`. It appends a protocol-formatted result and resumes generation; tools should return data rather than constructing protocol tokens themselves. Unknown functions and callable failures become tool errors. Malformed calls and simultaneous calls are rejected, and the CLI caps a user turn at five sequential calls.

For tools needing larger results or dependent sequences, first check context capacity and test the complete call/result/continuation flow. The current model and runtime limits still apply after adding a tool or swapping checkpoints.
