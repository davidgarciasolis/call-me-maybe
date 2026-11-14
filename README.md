*This project has been created as part of the 42 curriculum by davgarc4.*

# Call Me Maybe

## Description

Call Me Maybe translates natural language requests into JSON function calls.
The model chooses a function from the supplied definitions and extracts its
arguments. The program saves the call without executing the function.

For example, given `What is the sum of 2 and 3?`, the expected result is:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
}
```

The project uses `Qwen/Qwen3-0.6B` through the provided `llm_sdk`. During
generation, it checks candidate tokens against the structure of the available
functions. Instructions in the prompt describe the task; token restrictions
control the generated structure.

## Instructions

### Requirements and installation

- Python 3.10 or later.
- [uv](https://docs.astral.sh/uv/) for dependency management.
- The supplied `llm_sdk/` directory alongside `src/`.
- Internet access for the first model download, and storage for the model and
  its dependencies.

Run commands from the repository root:

```bash
uv sync
uv run python -m src
```

`pyproject.toml` declares NumPy, Pydantic and the local SDK as dependencies.
The SDK installs its own inference dependencies. Application code accesses the
model through the SDK's public methods.

### Command-line arguments

All three path arguments are optional:

| Argument | Default |
| --- | --- |
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

To use different files:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/my_results.json
```

To display help:

```bash
uv run python -m src --help
```

The program reports file, model and processing errors on standard error and
returns status `1`. Argument errors are handled by `argparse`, which returns
status `2`. A successful run returns `0`.

### Makefile

| Command | Purpose |
| --- | --- |
| `make install` | Install dependencies with `uv sync`. |
| `make run` | Run with Python 3.10 and the default file paths. |
| `make debug` | Run under Python's `pdb` debugger. |
| `make clean` | Remove Python bytecode and development caches. |
| `make lint` | Check `src` with flake8 and mypy, skipping imported modules. |

The Makefile defaults to `~/sgoinfre` for its virtual environment, caches and
temporary files. It configures two CPU threads through `OMP_NUM_THREADS` and
`MKL_NUM_THREADS`. For another storage location or thread count:

```bash
make install SGOINFRE=/path/to/storage
make run SGOINFRE=/path/to/storage THREADS=4
```

`make run` and `make debug` pass the default paths directly in their commands.
Use the `uv run python -m src` command for custom input or output paths.

The current lint target checks application code only. Subject IV.2 specifies
`flake8 .` and `mypy . --warn-return-any --warn-unused-ignores
--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`.
Passing the narrower checks does not establish compliance with those commands.

## Input and output

The function definitions file contains a JSON array. Each function has a
`name`, a `description`, a `parameters` object and a `returns` definition:

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "returns": {"type": "number"}
    }
]
```

Supported parameter types are `string`, `number` and `boolean`. Duplicate
function names and unsupported parameter types are rejected when reading the
definitions. Arrays, nested objects, `integer` and `null` parameters are not
currently supported. Return definitions describe the functions; the program
does not calculate their return values.

The requests file contains an array of objects with a string `prompt` field:

```json
[
    {"prompt": "What is the sum of 2 and 3?"},
    {"prompt": "What is the sum of 265 and 345?"}
]
```

The output is one JSON array, in request order. Each result contains exactly
`prompt`, `name` and `parameters`. The original request is copied into `prompt`.
The parent directories are created when necessary, and an existing output file
is overwritten. Results are written after all requests have completed, so a
generation failure prevents saving the new batch. Generated files under
`data/output/` are excluded from Git.

## Constrained decoding algorithm

The model and vocabulary are loaded once. Each vocabulary ID is decoded once
through the SDK and stored in a dictionary. Each request then follows these
steps:

1. Build a prompt containing the instructions, function definitions and request.
2. Encode the prompt into token IDs with the SDK.
3. Ask the SDK for next-token logits using the accumulated input IDs.
4. Copy the logits and find their maximum with `numpy.argmax()`.
5. Append that candidate's stored text to the accumulated response and check
   whether it is a valid prefix of a call to any available function.
6. If the candidate is invalid, set its copied logit to negative infinity and
   repeat the search. Accept the first valid candidate. Ties use the lowest ID.
7. Add the accepted ID to the context and its text to the response, then query
   the model again.
8. Stop when `json.loads()` parses the response as an object. Compare the
   concatenated token text with the SDK's decoding of all generated IDs.
9. Add the original request and collect the result for the output file.

Prefix validation starts from the beginning of the accumulated response for
each candidate. It requires compact JSON with `name` followed by `parameters`,
known function and parameter names, and parameters in definition order. It
checks separators, value syntax and closing braces while accepting incomplete
prefixes that can still be extended.

Numbers use decimal notation without exponents. Strings accept literal Unicode
characters and simple JSON escapes, but generated string arguments do not
support `\uXXXX` escapes. Structural names are matched against their
`json.dumps()` representation.

The selector checks candidates on demand rather than constructing a complete
mask for every vocabulary token. Invalid candidates receive negative infinity
in a working copy of the logits, leaving the original list unchanged.

## Design decisions and challenges

- **Small modules:** input validation, arguments, prompts, prefix rules, token
  selection and output writing have separate responsibilities.
- **Model-driven choice:** function selection and argument extraction come
  from the model's logits, without keyword rules for the example requests.
- **Simple grammar:** fixed key order and compact JSON reduce the number of
  structural variants the prefix validator must handle. The saved output is
  formatted with indentation for readability.
- **Public SDK interface:** tokenization, decoding and inference use only
  public SDK methods. The supplied SDK has not been changed for these features.
- **Separate request contexts:** requests share the loaded model and decoded
  vocabulary, but each has its own input and generated token IDs.
- **Token boundaries:** decoding each token separately keeps the code simple,
  but does not preserve UTF-8 bytes split across tokens. A final comparison with
  the SDK detects a mismatch; it does not recover the missing characters.
- **Error context:** processing errors include the request number. File and
  model failures are reported separately to help identify the failing step.

## Performance analysis

The subject targets at least 90% correct function and argument predictions,
100% parseable and schema-compliant results, and processing the test set in
under five minutes on standard hardware. These figures are assignment targets;
they are not benchmark results established for this implementation.

The program displays inference and selection durations per token, followed by
the total request-processing time. That total excludes model loading,
vocabulary preparation and output writing. To measure the whole command:

```bash
time uv run python -m src
```

Decoding the vocabulary once avoids repeated SDK decoding during selection.
However, every rejected candidate causes another maximum search and another
prefix check. Longer responses also increase validation work. The supplied SDK
processes the full input context on each inference call and exposes no public
cache for reusing previous inference states.

There is no generation token limit. Prefix validity alone does not guarantee
that generation will finish or that the extracted arguments are correct.

## Testing strategy

Development checks have covered prefix handling, token selection, error
reporting and integration with simulated SDK responses. The command-line
integration was checked with default paths, custom paths containing spaces,
output directory creation, help and invalid arguments. Makefile commands were
inspected with `make -n` without running inference.

Application code passed the following style and type checks:

```bash
uv run flake8 --jobs 1 src
uv run mypy src --warn-return-any --warn-unused-ignores \
    --ignore-missing-imports --disallow-untyped-defs \
    --check-untyped-defs --follow-imports=skip
```

These checks were run using the installed virtual environment. They match the
scope of the current lint target; the repository-wide commands required by
the subject were not run during this review. Development test scripts are
not included as a repository test suite.

The review also checked an existing file containing 11 results for expected
keys, request order, function names and parameter types. That file does not
establish current-model accuracy or runtime.

## Technical limitations

The implementation supports flat parameters of type `string`, `number` and
`boolean`, with the JSON syntax restrictions described above. Schema constraints
are enforced during prefix validation; there is no separate final schema
validation pass after parsing.

Token selection assumes candidate IDs exist in the decoded vocabulary. Empty
decoded tokens can leave the response unchanged, and isolated token decoding
can introduce replacement characters. Numbers outside the finite floating-point
range can fail when the output is written with `allow_nan=False`.

An empty request list still loads the model before saving an empty array.
Requests with no available functions are rejected before model loading.

## Project structure

```text
src/
    __main__.py       Application flow and generation
    argumentos.py    Command-line paths
    parseo.py        Input reading and validation
    modelo.py        Model and vocabulary loading
    prompts.py       Prompt construction
    decodificacion.py Token preparation and selection
    restricciones.py Function-call prefix validation
    validacion.py    Number, boolean and value validation
    salida.py        JSON output writing
    error.py         Application exceptions
llm_sdk/             Supplied model SDK
data/input/          Example definitions and requests
Makefile             Development commands
pyproject.toml       Dependencies and project metadata
uv.lock              Dependency lock file
```

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): reference for
  the model used by the SDK.
- [Python JSON documentation](https://docs.python.org/3/library/json.html):
  parsing and writing JSON.
- [Python argparse documentation](https://docs.python.org/3/library/argparse.html):
  command-line options and generated help.
- [NumPy argmax documentation](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html):
  maximum selection and tie behavior.
- [uv project guide](https://docs.astral.sh/uv/guides/projects/): dependency
  management and project execution.

### AI assistance

AI assisted with explaining the SDK and constrained decoding, proposing and
reviewing Python changes, simplifying validation and selection approaches,
writing temporary checks with simulated SDK responses, reviewing project
requirements, implementing command-line arguments, adjusting the Makefile and
drafting and checking this README against the implementation and subject.
Generated changes and explanations require review and understanding by the
author. Simulated checks do not replace real inference
measurements.
