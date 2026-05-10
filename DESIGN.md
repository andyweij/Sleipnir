# Sleipnir Design Notes

This document records the v0.1 design direction. It is intentionally concrete and small.

## Product Positioning

Sleipnir is a lightweight Agent SDK for self-hosted LLM platforms. It assumes the model
is exposed through an OpenAI-compatible API and focuses on single-agent tool use with
clear observability.

## Constraints

- Python 3.12+
- Runtime dependencies: `httpx`, `pydantic`, `rich`
- Package management: `uv`
- Testing: `pytest`
- Linting and formatting: `ruff`
- Synchronous API for v0.1
- ReAct loop
- Short-term memory only

## Public API Shape

The intended user-facing API starts with three concepts:

- `OpenAICompatibleClient`: talks to an OpenAI-compatible endpoint.
- `Agent`: owns the loop, tools, prompts, memory, and execution limits.
- `@agent.tool`: registers plain Python functions as callable tools.

Example:

```python
from sleipnir_agent import Agent, OpenAICompatibleClient

client = OpenAICompatibleClient(
    base_url="http://localhost:8000/v1",
    api_key="local",
    model="qwen2.5-7b-instruct",
)

agent = Agent(client=client, max_iterations=10)


@agent.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "sunny"


result = agent.run("Should I bring an umbrella in Taipei?")
print(result.content)
```

## Agent Loop

v0.1 uses a ReAct-style loop:

1. Build messages from system prompt, conversation history, and working memory.
2. Send request to the LLM with available tool schemas.
3. If the model returns a final answer, stop.
4. If the model returns tool calls, validate arguments and execute tools.
5. Append observations to memory.
6. Continue until final answer or `max_iterations`.

Safety controls:

- `max_iterations`, default 10
- optional token budget
- timeout and retry policy in the LLM client
- structured error events

## Tool Registration

Tools should be regular functions:

```python
@agent.tool
def search(query: str, max_results: int = 5) -> str:
    """Search for current information."""
    ...
```

The registry will inspect:

- function name
- docstring summary
- parameter names
- parameter type hints
- default values
- return value

Pydantic validates tool arguments before the function is called.

For v0.1, supported parameter types should stay narrow:

- `str`
- `int`
- `float`
- `bool`
- `list[str]`
- pydantic models only if they become necessary

## LLM Client

`OpenAICompatibleClient` wraps chat completion calls:

- non-streaming chat completion
- streaming chat completion
- OpenAI-style tool calling
- timeout
- retry with bounded attempts

The client should avoid hiding the wire format too aggressively. This project is meant
to be useful for self-hosted inference debugging, so raw request and response data should
remain inspectable.

## Memory

v0.1 memory is short-term only:

- conversation messages
- current task observations
- tool results

No vector database, embeddings, or long-term memory in v0.1.

## Observability

Each loop step should emit structured events. The first implementation can use in-memory
events plus optional Rich rendering.

Target event model:

```python
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    kind: Literal[
        "llm_request",
        "llm_response",
        "thought",
        "tool_call",
        "tool_result",
        "final_answer",
        "error",
    ]
    data: dict[str, Any] = Field(default_factory=dict)
```

## Error Handling

Errors should become visible trace events, not silent failures.

Initial categories:

- client timeout
- client HTTP error
- invalid tool arguments
- unknown tool name
- tool execution error
- max iterations reached

## Testing Strategy

Tests should avoid live LLM calls by default.

Initial test layers:

- schema generation from function signatures
- tool argument validation
- fake LLM client returning deterministic tool calls
- loop behavior for final answer, tool call, tool error, and max iterations

Live endpoint tests can be marked separately later.
