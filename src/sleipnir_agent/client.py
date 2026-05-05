"""OpenAI-compatible LLM client public API placeholder."""

from collections.abc import Generator
from pydantic import BaseModel
from typing import Protocol

# 1. Pydantic Models (等同於 Java 的 DTO)
class Message(BaseModel):
    role: str
    content: str

# 【Pythonic 慣用法】：使用 Protocol 定義介面，而不是 abc.ABC
class LLMClient(Protocol):
    def chat_completion(self, model: str, messages: list[Message], temperature: float = 0.7) -> str:
        ...

    def chat_completion_stream(self, model: str, messages: list[Message], temperature: float = 0.7) -> Generator[str, None, None]:
        ...

