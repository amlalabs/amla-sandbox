# amla-sandbox

This repository is the release source for the
[amla-sandbox](https://pypi.org/project/amla-sandbox/) Python package.
Development happens in
[the amlalabs monorepo](https://github.com/amlalabs/monorepo); this repo is
updated on release. The Rust runtime that compiles to `amla_sandbox.wasm`
lives in
[amla-sandbox-core](https://github.com/amlalabs/amla-sandbox-core); the exact
release tag this Python package was built against is recorded in
`.mirror-deps.json`.

amla-sandbox is a WASM sandbox with capability enforcement for AI agent code.
Agents can only call tools you explicitly provide, with constraints you
define. Sandboxed virtual filesystem. No network. No shell escape.

## Install

```sh
pip install amla-sandbox
```

No Docker. No VM. One binary, works everywhere.

## Quick start

```python
from amla_sandbox import create_sandbox_tool

sandbox = create_sandbox_tool()

# JavaScript
sandbox.run("console.log('hello'.toUpperCase())", language="javascript")
# Shell
sandbox.run("echo 'hello' | tr 'a-z' 'A-Z'", language="shell")

# With tools
def get_weather(city: str) -> dict:
    return {"city": city, "temp": 72}

sandbox = create_sandbox_tool(tools=[get_weather])
sandbox.run(
    "const w = await get_weather({city: 'SF'}); console.log(w);",
    language="javascript",
)
```

With capability constraints:

```python
from amla_sandbox import Sandbox, ToolCallCap, ConstraintSet, Param

sandbox = Sandbox(
    capabilities=[
        ToolCallCap(
            method_pattern="stripe/charges/*",
            constraints=ConstraintSet([
                Param("amount") <= 10000,
                Param("currency").is_in(["USD", "EUR"]),
            ]),
            max_calls=100,
        ),
    ],
    tool_handler=my_handler,
)
```

See the [PyPI page](https://pypi.org/project/amla-sandbox/) and the
`examples/` directory for the full API surface, framework integrations, and
the constraint DSL.

## Security model

The sandbox runs inside WebAssembly with WASI for a minimal syscall surface.
On top of WASM isolation, every tool call goes through capability validation;
access is explicitly granted, not implicitly available. See the
[Quick start](#quick-start) above and the upstream PyPI README for the full
explanation and tradeoffs.

## Building from source

For most users, installing from PyPI is recommended; the wheel includes the
prebuilt WASM binary. If you want to build the wheel yourself:

```sh
uv build
```

To regenerate the WASM artifact bundled inside the wheel, build it from
[amla-sandbox-core](https://github.com/amlalabs/amla-sandbox-core) at the tag
pinned in `.mirror-deps.json`, then drop the result at
`src/amla_sandbox/_wasm/amla_sandbox.wasm` before running `uv build`.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Pull requests against this mirror
will be clobbered on next release; please target the monorepo or open an issue
here.

## License

Python package code is MIT licensed. The bundled Rust WASM runtime is
AGPL-3.0-or-later OR BUSL-1.1.
