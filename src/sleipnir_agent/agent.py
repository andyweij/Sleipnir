"""Agent public API placeholders."""

from .client import LLMClient

class Agent:
    def __init__(self, client: LLMClient, model: str, tools: list | None = None):
        self.client = client
        self.model = model
        # Pythonic 慣用法：避免在參數預設值直接寫 mutable object (如 tools=[])
        # 應該預設為 None，然後在 __init__ 裡面給空列表
        self.tools = tools or []