"""OpenAI-compatible LLM client public API placeholder."""

from collections.abc import Generator
from typing import Protocol

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # 注意：OpenAI 格式中這裡通常是 JSON 字串


# 1. Pydantic Models (等同於 Java 的 DTO)
class Message(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # 當 role="tool" 時，用來對應是哪個 tool call 的結果
    name: str | None = None  # ← 加這行（tool role 需要對應 tool function name）


# 【Pythonic 慣用法】：使用 Protocol 定義介面，而不是 abc.ABC
class LLMClient(Protocol):
    def chat_completion(
        self, model: str, messages: list[Message], temperature: float = 0.7, tools: list[dict] | None = None
    ) -> Message: ...

    def chat_completion_stream(
        self, model: str, messages: list[Message], temperature: float = 0.7, tools: list[dict] | None = None
    ) -> Generator[str, None, None]: ...
