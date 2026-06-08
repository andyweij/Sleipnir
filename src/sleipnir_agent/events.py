import time
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
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)