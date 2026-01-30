# amla-sandbox

Every popular agent framework runs LLM-generated code via `subprocess` or `exec()`. That's arbitrary code execution on your host. One prompt injection and you're done.

| Framework | Execution Method                 | Source                                                                                                                                     |
| --------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| LangChain | `exec(command, globals, locals)` | [CVE-2025-68664](https://github.com/advisories/GHSA-c67j-w6g6-q2cm), [GitHub #5294](https://github.com/langchain-ai/langchain/issues/5294) |
| AutoGen   | `subprocess.run()`               | [Code Executors docs](https://microsoft.github.io/autogen/0.2/docs/tutorial/code-executors/)                                               |
| SWE-Agent | `subprocess.run(["bash", ...])`  | [SWE-ReX](https://github.com/SWE-agent/SWE-ReX)                                                                                            |

Some frameworks offer Docker isolation (OpenHands, AutoGen), but that requires running a Docker daemon and managing container infrastructure.

**amla-sandbox** is a WASM sandbox with capability enforcement. Agents can only call tools you explicitly provide, with constraints you define. Sandboxed virtual filesystem. No network. No shell escape.

```bash
uv pip install "git+https://github.com/amlalabs/amla-sandbox"
```

No Docker. No VM. One binary, works everywhere.

```python
from amla_sandbox import create_sandbox_tool

sandbox = create_sandbox_tool(tools=[stripe_api, database])

# Agent writes one script instead of 10 tool calls (JavaScript)
result = sandbox.run('''
    const txns = await stripe.listTransactions({customer: "cus_123"});
    const disputed = txns.filter(t => t.disputed);
    console.log(disputed[0]);
''', language="javascript")

# Or with shell pipelines
result = sandbox.run('''
    tool stripe.listTransactions --customer cus_123 | jq '[.[] | select(.disputed)] | .[0]'
''', language="shell")
```

## Why this matters

Tool-calling is expensive. Every MCP call is a round trip through the model:

```
LLM → tool → LLM → tool → LLM → tool → ...
```

Ten tool calls = ten LLM invocations. Code mode collapses this:

```
LLM → script that does all 10 things → result
```

But you can't just eval whatever the model spits out. So people either pay the token tax or run unsafe code. This gives you both: code-mode efficiency with actual isolation.

## Security model

The sandbox runs inside [WebAssembly](https://webassembly.org/docs/security/) with [WASI](https://wasi.dev/) for a minimal syscall interface. WASM provides memory isolation by design—linear memory is bounds-checked, and there's no way to escape to the host address space. The [wasmtime runtime](https://docs.wasmtime.dev/security.html) we use is built with defense-in-depth and has been [formally verified](https://www.usenix.org/conference/usenixsecurity22/presentation/bosamiya) for memory safety.

On top of WASM isolation, every tool call goes through capability validation:

```python
from amla_sandbox import Sandbox, MethodCapability, ConstraintSet, Param

sandbox = Sandbox(
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

# This works
sandbox.execute('await stripe.charges.create({amount: 500, currency: "USD"})')

# This fails - amount exceeds capability
sandbox.execute('await stripe.charges.create({amount: 50000, currency: "USD"})')
```

The design draws from [capability-based security](https://en.wikipedia.org/wiki/Capability-based_security) as implemented in systems like [seL4](https://sel4.systems/)—access is explicitly granted, not implicitly available. Agents don't get ambient authority just because they're running in your process. This matters because prompt injection is a [fundamental unsolved problem](https://simonwillison.net/2025/Apr/11/prompt-injection-mitigation/); defense in depth through capability restriction limits the blast radius.

## Quick start

```python
from amla_sandbox import create_sandbox_tool

sandbox = create_sandbox_tool()

# JavaScript
sandbox.run("console.log('hello'.toUpperCase())", language="javascript")  # -> "HELLO"

# Shell
sandbox.run("echo 'hello' | tr 'a-z' 'A-Z'", language="shell")  # -> "HELLO"

# With tools
def get_weather(city: str) -> dict:
    return {"city": city, "temp": 72}

sandbox = create_sandbox_tool(tools=[get_weather])
sandbox.run("const w = await get_weather({city: 'SF'}); console.log(w);", language="javascript")
```

With constraints:

```python
sandbox = create_sandbox_tool(
    tools=[transfer_money],
    constraints={
        "transfer_money": {
            "amount": "<=1000",
            "currency": ["USD", "EUR"],
        },
    },
    max_calls={"transfer_money": 10},
)
```

## JavaScript API Notes

**Tools require object syntax:**

```javascript
// WORKS - tools always take an object argument
await get_weather({city: "SF"});
await transfer({to: "alice", amount: 500});

// FAILS - positional arguments don't work
await get_weather("SF");  // Error: argument after ** must be a mapping
```

**Use `return` or `console.log()` for output:**

```javascript
// Return value is captured and output
return await get_weather({city: "SF"});  // -> {"city":"SF","temp":72}
return {a: 1, b: 2};  // -> {"a":1,"b":2}
return "hello";  // -> hello (strings not double-quoted)

// console.log also works
console.log(JSON.stringify({a: 1}));  // -> {"a":1}

// No return = no output
const x = 42;  // -> (no output)
```

**VFS is writable only under /workspace and /tmp:**

```javascript
// WORKS - /workspace and /tmp are ReadWrite
await fs.writeFile('/workspace/data.json', '{}');
await fs.mkdir('/tmp/cache');

// FAILS - root is read-only
await fs.mkdir('/mydir');  // EACCES: Permission denied
```

## LangGraph

For LangGraph integration:

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from amla_sandbox import create_sandbox_tool

sandbox = create_sandbox_tool(tools=[get_weather, search_db])
agent = create_react_agent(
    ChatAnthropic(model="claude-sonnet-4-20250514"),
    [sandbox.as_langchain_tool()]  # LLM writes JS/shell that calls your tools
)
```

For fine-grained capability control:

```python
from amla_sandbox import SandboxTool, MethodCapability, ConstraintSet, Param

caps = [
    MethodCapability(
        method_pattern="mcp:search_db",
        constraints=ConstraintSet([Param("query").starts_with("SELECT")]),
        max_calls=5,
    )
]

sandbox_tool = SandboxTool.from_functions([search_db], capabilities=caps)
agent = create_react_agent(model, [sandbox_tool.as_langchain_tool()])
```

## Architecture

```
┌────────────────────────────────────────────────┐
│              WASM Sandbox                      │
│  ┌──────────────────────────────────────────┐  │
│  │         Async Scheduler                  │  │
│  │   tasks waiting/running/ready            │  │
│  └──────────────────────────────────────────┘  │
│  ┌────────────┐ ┌──────────┐ ┌──────────────┐  │
│  │  VFS       │ │ Shell    │ │ Capabilities │  │
│  │ /workspace │ │ builtins │ │ validation   │  │
│  └────────────┘ └──────────┘ └──────────────┘  │
│                    ↓ yield                     │
└════════════════════════════════════════════════┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│              Python Host                    │
│                                             │
│   while sandbox.has_work():                 │
│       req = sandbox.step()  # tool call     │
│       sandbox.resume(execute(req))          │
│                                             │
└─────────────────────────────────────────────┘
```

The sandbox yields on tool calls. Host executes them (after capability checks) and resumes. [QuickJS](https://bellard.org/quickjs/) runs inside WASM for the JS runtime.

## Precompilation

First run compiles the WASM module (~300ms). Cache it:

```bash
amla-precompile
```

Subsequent loads: ~0.5ms.

## Constraint DSL

```python
from amla_sandbox import Param, ConstraintSet

constraints = ConstraintSet([
    Param("amount") >= 100,
    Param("amount") <= 10000,
    Param("currency").is_in(["USD", "EUR"]),
    Param("path").starts_with("/api/"),
])
```

Pattern matching for method names:

- `stripe/charges/create` — exact match
- `stripe/charges/*` — single path segment
- `stripe/**` — zero or more segments

## Tradeoffs

**What you get:** Isolation without infrastructure. Capability enforcement. Token efficiency.

**What you don't get:** Full Linux environment. Native module support. GPU access. Infinite loop protection (a `while(true){}` will hang - the step limit only counts WASM yields, not JS instructions).

If you need a real VM with persistent state and arbitrary dependencies, use [e2b](https://e2b.dev) or [Modal](https://modal.com). amla-sandbox is for the common case: agents running generated code with controlled tool access.

## License

Python code is MIT. The WASM binary is proprietary—you can use it with this package but can't extract or redistribute it separately.

---

[Website](https://amlalabs.com/sandbox) · [Examples](./examples) · [Docs](./docs)
