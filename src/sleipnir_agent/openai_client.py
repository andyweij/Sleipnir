import httpx

from .client import LLMClient, Message, ToolCall


class OpenAILikeClient(LLMClient):
    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 60.0):
        # 防呆：確保 base_url 結尾沒有斜線
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # 【Pythonic 慣用法】: 不要每次請求都新建 HTTP Client。
        # 建立一個實例層級的 httpx.Client 可以重複使用底層的 TCP 連線 (Connection Pooling)。
        self._http_client = httpx.Client(
            base_url=self.base_url, timeout=self.timeout, headers={"Authorization": f"Bearer {self.api_key}"}
        )

    def chat_completion(
        self, model: str, messages: list[Message], temperature: float = 0.7, tools: list[dict] | None = None
    ) -> Message:
        payload = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = self._http_client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        msg_data = data["choices"][0]["message"]

        # 判斷是不是 tool call
        if msg_data.get("tool_calls"):
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
                    for tc in msg_data["tool_calls"]
                ],
            )

        return Message(role="assistant", content=msg_data.get("content", ""))
