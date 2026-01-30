# amla-sandbox Examples

Curated examples demonstrating amla-sandbox capabilities. Start with `quick_start.py` and explore based on your needs.

## Getting Started

```bash
uv pip install "git+https://github.com/amlalabs/amla-sandbox"

# Run any example
python examples/quick_start.py
```

## Examples

### Core API

| Example | Description |
|---------|-------------|
| [quick_start.py](quick_start.py) | Hello world - shell commands, JavaScript, basic tools |
| [tools.py](tools.py) | Define tools from Python functions |
| [async_tools.py](async_tools.py) | Async tool handlers for I/O-bound operations |
| [constraints.py](constraints.py) | Constraint DSL reference (`Param`, numeric, membership, string) |
| [capabilities.py](capabilities.py) | MethodCapability patterns and authorization |
| [rate_limiting.py](rate_limiting.py) | Call budgets and rate limiting |
| [framework_ingestion.py](framework_ingestion.py) | Import tools from LangChain, OpenAI, Anthropic |

### Features

| Example | Description |
|---------|-------------|
| [shell_pipelines.py](shell_pipelines.py) | Data processing with grep, jq, sort, uniq |
| [vfs_operations.py](vfs_operations.py) | Virtual filesystem read/write/persist |
| [langgraph_agent.py](langgraph_agent.py) | LangGraph integration with create_react_agent |
| [langgraph_codeact.py](langgraph_codeact.py) | CodeAct agent pattern with LangGraph |
| [langgraph_openai.py](langgraph_openai.py) | LangGraph with OpenAI GPT models |
| [streaming.py](streaming.py) | Real-time output callbacks |
| [error_handling.py](error_handling.py) | Graceful error recovery patterns |
| [audit_logging.py](audit_logging.py) | Compliance logging for enterprise |

### Real-World Patterns

| Example | Description |
|---------|-------------|
| [customer_support.py](customer_support.py) | Complete support agent with tiered permissions |
| [data_pipeline.py](data_pipeline.py) | ETL data processing pattern |
| [multi_tenant.py](multi_tenant.py) | Enterprise multi-tenant isolation |
| [insurance_claims.py](insurance_claims.py) | Multi-agent workflow with delegation |

### Reference

| Example | Description |
|---------|-------------|
| [recipes.py](recipes.py) | Copy-paste snippets for common tasks |
| [testing.py](testing.py) | Unit and integration testing patterns |

## Additional Resources

- **[agents/](agents/)** - Complete agent implementations (code reviewer, data analyst, research assistant)

## Learning Path

1. **New to amla-sandbox?** Start with `quick_start.py`, then `tools.py` and `shell_pipelines.py`
2. **Building agents?** See `langgraph_agent.py`, `async_tools.py`, and `error_handling.py`
3. **Going to production?** Check `audit_logging.py`, `rate_limiting.py`, and `multi_tenant.py`
4. **Migrating from other frameworks?** See `framework_ingestion.py`
