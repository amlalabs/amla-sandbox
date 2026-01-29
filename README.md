# amla-sandbox

Lightweight code execution for AI agents. Reduces token costs by 98%. Works everywhere.

```python
from amla_sandbox import create_bash_tool

bash = create_bash_tool()
print(bash.run("echo 'Hello from the sandbox!'"))
# Output: Hello from the sandbox!
```

## The Problem

If you're running LLM-generated code through subprocess, you're running eval on your system.

Most popular agent frameworks do exactly this:

| Framework | Execution Method | Source |
|-----------|-----------------|--------|
| LangChain | Direct Python exec | [Security warning](https://python.langchain.com/docs/security/) |
| AutoGPT | Shell subprocess | [Source](https://github.com/Significant-Gravitas/AutoGPT) |
| AutoGen | Python subprocess | [Docs](https://microsoft.github.io/autogen/0.2/docs/tutorial/code-executors/) |
| SWE-Agent | Bash subprocess | [Docs](https://github.com/SWE-agent/SWE-agent) |
| MetaGPT | Python subprocess | [Source](https://github.com/FoundationAgents/MetaGPT) |

That's arbitrary code execution on your host. No capability restrictions. One prompt injection away from disaster.

Meanwhile, you're also paying for it—badly. Every MCP tool call is a round trip:

```
LLM → tool_call → LLM → tool_call → LLM → tool_call → LLM → ...
      (10 round trips = 10 LLM calls, massive token overhead)
```

Code mode collapses this to one call:

```
LLM → script that does all 10 things → result
      (1 round trip, 98% fewer tokens)
```

## The Solution

amla-sandbox gives you code mode efficiency with actual capability boundaries:

```bash
pip install amla-sandbox
```

That's it. No Docker, no VMs, no infrastructure. One WASM binary that works on macOS, Linux, and Windows.

```python
from amla_sandbox import create_bash_tool

# Your existing tools, now with code execution
bash = create_bash_tool(tools=[stripe_api, database])

# Agent writes one script instead of 10 tool calls
result = bash.run('''
    const txns = await stripe.listTransactions({customer: "cus_123"});
    await fs.writeFile('/workspace/txns.json', JSON.stringify(txns));
    const disputed = shell("grep 'disputed' /workspace/txns.json | head -1");
    console.log(disputed);
''')
```

The agent can only call tools you provide. No filesystem access. No network access. No shell escape. Just your tools, through a capability-checked chokepoint.

## Core APIs

- `create_bash_tool(tools=...)` — dead-simple shell+tool surface, great default
- `SandboxTool.from_functions([...])` — LangGraph-ready tool with explicit capabilities
- `create_amla_codeact` / `create_amla_sandbox` — CodeAct helpers if you prefer that flow
- `Sandbox` — low-level access if you need to manage the runtime directly
- `precompile_module()` — optional startup optimization (300ms → 0.5ms)

## Quick Start

```bash
pip install amla-sandbox

# Or with framework support
pip install "amla-sandbox[langgraph]"  # LangGraph + CodeAct helpers
pip install "amla-sandbox[codeact]"    # CodeAct-only helper
```

The simplest way to give agents code execution:

```python
from amla_sandbox import create_bash_tool

# Shell only - no external tools
bash = create_bash_tool()
output = bash.run("echo 'Hello' | tr 'a-z' 'A-Z'")

# With Python functions as tools
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return {"city": city, "temp": 72}

bash = create_bash_tool(tools=[get_weather])

# Call tools from JavaScript
bash.run("const w = await get_weather({city: 'SF'}); console.log(w);")

# Or use shell syntax
bash.run("tool get_weather --city SF")
```

### With Constraints

```python
bash = create_bash_tool(
    tools=[transfer_money, get_weather],
    constraints={
        "transfer_money": {
            "amount": "<=1000",       # Numeric bounds
            "currency": ["USD", "EUR"],  # Allowed values
        },
    },
    max_calls={"transfer_money": 10},  # Call limits
)
```

## Precompilation (Optional)

First sandbox creation compiles the WASM module (~300ms). Precompile for instant startup:

```bash
# CLI: precompile and cache to disk
amla-precompile

# Or programmatically
from amla_sandbox import precompile_module
precompile_module()  # ~0.5ms on subsequent loads
```

The cache is stored in `~/.cache/amla-sandbox/` and includes the wasmtime version and platform in the key for compatibility.

## Examples

See [examples/](examples/) for 20 curated examples:

- **quick_start.py** - Hello world and basic usage
- **tools.py** - Define tools from Python functions
- **langgraph_agent.py** - Building AI agents with LangGraph
- **constraints.py** - Parameter validation patterns
- **customer_support.py** - Complete real-world agent example

## Framework Integration

### LangGraph (Recommended)

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from amla_sandbox import create_bash_tool

bash = create_bash_tool(tools=[get_weather, search_db])
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

agent = create_react_agent(model, [bash.as_langchain_tool()])
result = agent.invoke({"messages": [("user", "What's the weather in SF?")]})
```

### LangGraph (Custom capabilities)

```python
from langgraph.prebuilt import create_react_agent
from amla_sandbox import (
    SandboxTool,
    ConstraintSet,
    MethodCapability,
    Param,
)

def search_db(query: str) -> dict:
    return {"results": [query]}

caps = [
    MethodCapability(
        method_pattern="mcp:search_db",
        constraints=ConstraintSet([Param("query").starts_with("SELECT")]),
        max_calls=5,
    )
]

sandbox_tool = SandboxTool.from_functions(
    [search_db],
    capabilities=caps,
)

agent = create_react_agent(model, [sandbox_tool.as_langchain_tool()])
result = agent.invoke({"messages": [("user", "Search for SELECT status")]})
```

## Capability Enforcement

All tool calls are validated against capabilities before execution:

```python
from amla_sandbox import Sandbox, MethodCapability, ConstraintSet, Param

sandbox = Sandbox(
    pca=pca_bytes,
    capabilities=[
        MethodCapability(
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

# This call is allowed
sandbox.execute('await stripe.charges.create({amount: 500, currency: "USD"})')

# This call fails with CapabilityError - amount exceeds limit
sandbox.execute('await stripe.charges.create({amount: 50000, currency: "USD"})')
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WASM Sandbox                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 Async Scheduler                        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐               │  │
│  │  │ Task 1  │  │ Task 2  │  │ Task 3  │  ...          │  │
│  │  │ waiting │  │ running │  │ ready   │               │  │
│  │  │ on tool │  │         │  │         │               │  │
│  │  └─────────┘  └─────────┘  └─────────┘               │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │   VFS    │ │  Shell   │ │ QuickJS  │ │ Capabilities │  │
│  │ /tmp/    │ │ grep,cat │ │ async/   │ │  validation  │  │
│  │ /work/   │ │ cut,sort │ │ await    │ │  per call    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│                          ↓ yield                           │
└════════════════════════════════════════════════════════════┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python Host                               │
│                                                              │
│   while sandbox.has_work():                                  │
│       request = sandbox.step()   # tool call, sleep, I/O     │
│       result = execute(request)  # your MCP server, API      │
│       sandbox.resume(result)                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Pattern Matching

The capability system supports glob patterns:

- `stripe/charges/create` - exact match
- `stripe/charges/*` - matches single path segment
- `stripe/**` - matches zero or more segments

## Tool Name Transformation

When tools are exposed in the JavaScript sandbox, their names are transformed
to valid JS identifiers:

| Python Tool Name | JavaScript Function |
|-----------------|---------------------|
| `weather/current` | `weather_current()` |
| `stripe.listTransactions` | `stripe_listTransactions()` |
| `my-tool` | `my_tool()` |
| `foo:bar` | `foo_bar()` |

**Rule:** Dots (`.`), slashes (`/`), dashes (`-`), colons (`:`) → underscores (`_`)

## Constraint DSL

Use the `Param` builder for ergonomic constraint creation:

```python
from amla_sandbox import Param, ConstraintSet

constraints = ConstraintSet([
    Param("amount") >= 100,       # Greater than or equal
    Param("amount") <= 10000,     # Less than or equal
    Param("currency").is_in(["USD", "EUR"]),  # Set membership
    Param("path").starts_with("/api/"),       # String prefix
])
```

## Persisting State Between Executions

The VFS persists across `execute()` calls within the same Sandbox instance.
Use `/workspace/` for read/write operations:

```javascript
// First execution - save state
const result = await expensive_computation();
await fs.writeFile('/workspace/cache.json', JSON.stringify(result));
console.log("Computed");

// Second execution - restore state
const cached = JSON.parse(await fs.readFile('/workspace/cache.json'));
console.log("Using cached:", cached.length, "items");
```

## Why amla-sandbox

### 1. Token Efficiency (98% reduction)

| Tool Mode (10 round trips) | Code Mode (1 round trip) |
|---------------------------|-------------------------|
| LLM → tool → LLM → tool → ... | LLM → script → result |
| $0.50 for a multi-step task | $0.03 for the same task |
| Context fills with raw responses | Only final results enter context |

### 2. Security by Default

| Raw Subprocess | amla-sandbox |
|---------------|--------------|
| Arbitrary code execution on host | WASM sandbox, no syscalls |
| No capability restrictions | Every tool call validated |
| Full filesystem/network access | Only your tools, nothing else |
| One prompt injection = game over | Agent can only do what you allow |

### 3. Zero Infrastructure

| Other Sandboxes | amla-sandbox |
|-----------------|--------------|
| Docker: daemon, images, networking | `pip install amla-sandbox` |
| Firecracker: KVM, root, Linux only | Single WASM binary |
| E2B/Modal: API keys, per-exec cost | Works offline, no API |

## Distribution

This package bundles a pre-compiled WASM runtime binary. The Python integration layer is open source; the WASM binary is proprietary.

```
amla-sandbox/
├── src/amla_sandbox/     # Open source Python code
│   ├── capabilities/     # Capability enforcement
│   ├── runtime/          # wasmtime integration
│   ├── tools/            # Tool definitions + ingestion helpers
│   ├── langgraph.py      # LangGraph/CodeAct integration
│   ├── bash_tool.py      # AI SDK-style ergonomics
│   └── _wasm/
│       └── amla_sandbox.wasm  # Proprietary binary (Git LFS)
└── tests/                # Open source tests
```

## License

This project uses a dual-license model:

- **Python code** (everything in `src/amla_sandbox/` except `_wasm/`): [MIT License](LICENSE)
- **WASM runtime** (`src/amla_sandbox/_wasm/amla_sandbox.wasm`): Proprietary, owned by Amla Labs

You are free to use, modify, and distribute the Python code under the MIT license. The bundled WASM binary is provided for use with this package but may not be extracted, reverse-engineered, or distributed separately.

Copyright (c) 2025 Amla Labs
