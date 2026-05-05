import json
import httpx
from collections.abc import Generator
from .client import Message

class GeminiClient():
    def __init__(self, api_key: str, timeout: float = 60.0):
        self.api_key = api_key
        # Gemini REST API Base URL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self._http_client = httpx.Client(timeout=timeout)

    def _format_payload(self, messages: list[Message], temperature: float) -> dict:
        """將統一的 Message 轉換為 Gemini REST API 格式"""
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                # 轉換 role 名稱: assistant -> model
                role = "model" if msg.role == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature}
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
            
        return payload

    def chat_completion_stream(self, model: str, messages: list[Message], temperature: float = 0.7) -> Generator[str, None, None]:
        # Gemini 的 Streaming endpoint 格式
        url = f"{self.base_url}/{model}:streamGenerateContent?key={self.api_key}&alt=sse"
        payload = self._format_payload(messages, temperature)

        # 這裡同樣使用 httpx 的 streaming
        with self._http_client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            # Gemini 的 SSE 格式與 OpenAI 略有不同，需要稍微處理 JSON parsing
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        # 安全取值，提取 Gemini 的回應 text
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                yield parts[0].get("text", "")
                    except json.JSONDecodeError:
                        continue
                        
    def chat_completion(self, model: str, messages: list[Message], temperature: float = 0.7) -> str:
        # 非串流實作 (概念類似，打 generateContent endpoint)
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        response = self._http_client.post(url, json=self._format_payload(messages, temperature))
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]