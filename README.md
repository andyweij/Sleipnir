# Sleipnir Agent SDK

Sleipnir is a lightweight Agent SDK for self-hosted LLMs.

It is designed for teams that run their own OpenAI-compatible inference endpoints,
including vLLM, llama.cpp servers, Qualcomm Genie-backed services, or internal LLM
platforms. The goal is to provide a small, understandable agent loop with first-class
tool calling and traceability, without pulling in a large agent framework.

> Status: v0.0.1 planning scaffold. The public API below is the intended shape, not a
> complete implementation yet.

## Why Sleipnir

Most agent frameworks are built around broad abstractions: chains, graphs, retrievers,
multi-agent orchestration, hosted model assumptions, and fast-moving plugin ecosystems.
Those are powerful, but they can be too much when the real need is:

- call a self-hosted model through an OpenAI-compatible API
- register plain Python functions as tools
- let the model decide when to call a tool
- keep every thought, action, and observation traceable
- stop safely with iteration limits and clear error handling

Sleipnir starts from that smaller surface area.

## Design Goals

- Self-hosted first: works naturally with OpenAI-compatible local or private endpoints.
- Minimal dependencies: only `httpx`, `pydantic`, and `rich` at runtime.
- Pythonic tool registration: use decorators and type hints instead of framework objects.
- Observable by default: expose each reasoning step as structured trace events.
- ReAct loop first: simple step-by-step reasoning, action, observation, and final answer.
- Sync first: v0.1 focuses on a clear synchronous API; async can come later.

## Non Goals for v0.1

- No RAG.
- No multi-agent orchestration.
- No GUI.
- No vector memory or long-term memory.
- No dependency on LangChain, LlamaIndex, AutoGen, CrewAI, or similar frameworks.

## Quick Start

Install with `uv`:

```bash
uv add sleipnir_agent
```

Create an agent connected to an OpenAI-compatible endpoint:

```python
from sleipnir_agent import Agent, OpenAICompatibleClient

client = OpenAICompatibleClient(
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local",
    model="qwen2.5-7b-instruct",
    timeout=30.0,
)

agent = Agent(
    client=client,
    system_prompt="You are a precise assistant. Use tools when they help.",
    max_iterations=10,
)


@agent.tool
def multiply(left: int, right: int) -> int:
    """Multiply two integers."""
    return left * right


answer = agent.run("What is 17 * 23? Use the multiply tool.")
print(answer.content)
```

Expected shape:

```text
391
```

## Tool Calling

Tools are ordinary Python functions. Sleipnir reads the function signature, type hints,
and docstring to produce an OpenAI function-calling schema.

```python
@agent.tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information."""
    ...
```

The generated schema should look conceptually like this:

```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web for current information.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string" },
        "max_results": { "type": "integer", "default": 5 }
      },
      "required": ["query"]
    }
  }
}
```

## Trace Events

Sleipnir should make the agent loop easy to inspect.

```python
result = agent.run("Find the latest stable Python version and summarize it.")

for event in result.trace:
    print(event.kind, event.data)
```

Possible event kinds:

- `llm_request`
- `llm_response`
- `thought`
- `tool_call`
- `tool_result`
- `final_answer`
- `error`

## Project Layout

```text
.
├── .github/workflows/ci.yml
├── examples/
│   ├── basic_chat.py
│   ├── tool_calling.py
│   └── react_loop.py
├── src/
│   └── sleipnir_agent/
│       ├── __init__.py
│       ├── agent.py
│       ├── cli.py
│       ├── client.py
│       ├── events.py
│       ├── memory.py
│       ├── tools.py
│       └── py.typed
├── tests/
│   └── test_public_api_placeholder.py
├── DESIGN.md
├── README.md
└── pyproject.toml
```

## Roadmap

### v0.0.1

- Project scaffold with `uv`.
- README vision and quick start.
- DESIGN.md decision record.
- Public API examples.
- CI with `ruff` and `pytest`.

### v0.1

- OpenAI-compatible sync client.
- Streaming support.
- Tool registration with automatic schema generation.
- Single tool-call execution path.
- ReAct loop with `max_iterations` and early stopping.
- Structured trace events.
- Basic web search tool with a pluggable search backend.

### v0.2

- Async API.
- Long-term memory.
- Additional search backends such as self-hosted SearXNG.
- Optional multi-step planning experiments.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run ruff format .
uv run pytest
```

## License

MIT
