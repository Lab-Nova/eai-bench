"""Registering the endpoint with `bfcl-eval` without writing to site-packages.

`bfcl-eval` has no plugin mechanism: MODEL_CONFIG_MAPPING is three static dict
literals merged in `bfcl_eval/constants/model_config.py`, and the upstream instruction
for adding a model is to edit that file. The previous incarnation of this benchmark did
exactly that -- `cat >> .../site-packages/bfcl_eval/constants/model_config.py` -- which
does not survive a reinstall and leaves no record of what was registered.

Registering in-process works instead because every consumer does
`from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING`, binding the *same*
dict object, and every lookup happens at call time rather than import time. Mutating
that dict before invoking the CLI is therefore visible everywhere it matters.
"""

import os
import re
import sys


def registry_name(model):
    """The name BFCL knows the model by. Must contain no '_' and no '/'.

    Both restrictions are load-bearing:
      * base_handler does `registry_name.replace("/", "_")` to name result/ and score/,
      * eval_runner reverses it with `model_name.replace("_", "/")` to turn that
        directory name back into a registry key.
    A name containing '_' therefore fails to look itself up at evaluation time.
    """
    safe = re.sub(r"[^A-Za-z0-9.-]", "-", model)
    return f"{safe}-sglang-FC"


def write_env(project_root, endpoint, api_key="EMPTY"):
    """Write $BFCL_PROJECT_ROOT/.env -- the only place the endpoint can be set.

    Both `bfcl generate` and `bfcl evaluate` call
    `load_dotenv(dotenv_path=DOTENV_PATH, ..., override=True)`. Because override is
    True, this file BEATS anything exported in the shell: an OPENAI_BASE_URL exported
    into the environment is silently discarded the moment a .env exists. Writing it
    here also binds the endpoint physically to the results directory.
    """
    os.makedirs(project_root, exist_ok=True)
    path = os.path.join(project_root, ".env")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"OPENAI_BASE_URL={endpoint}\n")
        # The openai client constructor raises on an empty key, so it must be non-empty
        # even though a local sglang server ignores it.
        f.write(f"OPENAI_API_KEY={api_key or 'EMPTY'}\n")
    return path


def register(model, project_root, name=None):
    """Insert the endpoint's ModelConfig into the live mapping. Idempotent."""
    if os.environ.get("BFCL_PROJECT_ROOT") != str(project_root):
        raise RuntimeError("BFCL_PROJECT_ROOT must be set before register() is called: "
                           "importing bfcl_eval creates result/ and score/ under it")
    from bfcl_eval.constants import model_config as mc
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler)

    name = name or registry_name(model)
    mc.MODEL_CONFIG_MAPPING[name] = mc.ModelConfig(
        model_name=model,           # the string sent as "model" in the request body
        display_name=f"{model} (sglang, FC)",
        url="", org="", license="",
        model_handler=OpenAICompletionsHandler,
        input_price=None, output_price=None,
        is_fc_model=True,
        # True if the model does not support '.' in function names. This is used by the
        # checker only, so it never requires regeneration -- but setting it False
        # silently zeroed two categories on an earlier run. Leave it alone.
        underscore_to_dot=True,
    )
    return name


def run_cli(argv, project_root, model, name=None):
    """Register, then invoke the `bfcl` CLI in-process with `argv`."""
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    name = register(model, project_root, name)
    from bfcl_eval.__main__ import cli
    saved = sys.argv
    sys.argv = ["bfcl", *argv]
    try:
        cli()
    except SystemExit as e:  # typer exits 0 on success; anything else is a real failure
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = saved
    return name
