# Sleipnir

> *In Norse mythology, Sleipnir is Odin's eight-legged steed—
> the fastest horse, capable of traversing all nine worlds.*

A lightweight agent framework for self-hosted LLMs, 
designed to traverse multiple tools and data sources 
with autonomous decision-making.

## Why Sleipnir?

- **Eight legs**: built-in support for multiple tools 
  (web search, code execution, RAG, custom)
- **Nine worlds**: works across self-hosted runtimes 
  (vLLM, llama.cpp, Qualcomm NPU)
- **Agent-in-the-loop**: autonomously decides when to search, 
  when to stop, when to ask


## Why this exists

Most agent frameworks (LangChain, LlamaIndex) are designed for OpenAI / Anthropic APIs.
They're heavy, change frequently, and assume you have unlimited cloud budget.

This framework is different:
- Built for **self-hosted models** (vLLM, llama.cpp, Qualcomm NPU)
- Minimal dependencies (just `httpx` and `pydantic`)
- Full **observability** - every reasoning step is traceable
- **Agent-in-the-loop**: the agent autonomously decides when to search,
  when to stop, when to ask for clarification

## Status

🚧 In active development. Targeting v0.1 by [date].

Getting Started
(Coming soon: Installation and Quick Start Guide)

## Roadmap

- [x] Core LLM client (OpenAI-compatible)
- [ ] Tool registration via `@tool` decorator
- [ ] ReAct loop
- [ ] Web search tool with multi-backend support
- [ ] Self-evaluation loop
- [ ] Query refinement
