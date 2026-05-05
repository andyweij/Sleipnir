
import json

import httpx
from collections.abc import Generator
from .client import Message

class OpenAILikeClient():
    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 60.0):
        # 防呆：確保 base_url 結尾沒有斜線
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # 【Pythonic 慣用法】: 不要每次請求都新建 HTTP Client。
        # 建立一個實例層級的 httpx.Client 可以重複使用底層的 TCP 連線 (Connection Pooling)。
        self._http_client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    def chat_completion(self, model: str, messages: list[Message], temperature: float = 0.7) -> str:
        """非串流 (Non-streaming) 的基本呼叫"""
        payload = {
            "model": model,
            # 【Pythonic 慣用法】: List Comprehension，取代 Java 的 for-loop add 或 stream().map()
            "messages": [m.model_dump() for m in messages], 
            "temperature": temperature,
            "stream": False
        }
        
        response = self._http_client.post("/chat/completions", json=payload)
        response.raise_for_status()  # 如果不是 2xx，直接拋出例外
        
        # 快速解析並回傳文字
        return response.json()["choices"][0]["message"]["content"]
        
    def chat_completion_stream(self, model: str, messages: list[Message], temperature: float = 0.7) -> Generator[str, None, None]:
        """串流 (Streaming) 呼叫 - v0.1 Agent 核心所需"""
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": True
        }
        
        # 【Pythonic 慣用法】: Context Manager (with 語句)
        # 等同於 Java 的 try-with-resources，確保串流讀取完畢後正確關閉 Response 資源
        with self._http_client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        # dict.get() 是安全取值的好習慣，避免 KeyError
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            # 【Pythonic 慣用法】: Yield 產生 Generator
                            # 讓上層呼叫者可以用 for token in client.chat_completion_stream(...) 逐步接收
                            yield delta
                    except json.JSONDecodeError:
                        continue