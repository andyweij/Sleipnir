import json
import uuid
from collections.abc import Generator

import httpx

from .client import LLMClient, Message, ToolCall


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, timeout: float = 60.0):
        self.api_key = api_key
        # Gemini REST API Base URL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self._http_client = httpx.Client(timeout=timeout)

    def _format_payload(self, messages: list[Message], temperature: float, tools: list[dict] | None = None) -> dict:
        """將標準的 Message 與 OpenAI Schema 轉換為 Gemini REST API 格式"""
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role == "tool":
                # 將工具執行結果餵回給 Gemini
                contents.append({
                    "role": "user", # Gemini 沒有 tool role，通常放 user 或 function_response
                    "parts": [{"functionResponse": {
                        "name": msg.name if hasattr(msg, 'name') else "tool_response", 
                        "response": {"result": msg.content}
                    }}]
                })
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content or ""}] # 防呆處理 content 為 None 的情況
                })

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature}
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
            
        # 轉換 Tools Schema (OpenAI -> Gemini)
        if tools:
            gemini_tools = []
            for t in tools:
                gemini_tools.append({
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "parameters": t["function"]["parameters"]
                })
            payload["tools"] = [{"function_declarations": gemini_tools}]
            
        return payload


    def chat_completion(
        self, 
        model: str, 
        messages: list[Message], 
        temperature: float = 0.7,
        tools: list[dict] | None = None
    ) -> Message:
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        response = self._http_client.post(url, json=self._format_payload(messages, temperature, tools))
        response.raise_for_status()
        
        data = response.json()
        
        parts = data["candidates"][0]["content"]["parts"]
        
        # 判斷 Gemini 是否決定呼叫工具
        if "functionCall" in parts[0]:
            fc = parts[0]["functionCall"]
            # 把它包裝成 OpenAI 的 ToolCall 格式返回
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}", # 產生一組假的 ID，因為 Gemini 不給 ID
                        name=fc["name"],
                        arguments=json.dumps(fc.get("args", {}))
                    )
                ]
            )
            
        # 如果不是工具呼叫，就是一般文字回覆
        return Message(
            role="assistant",
            content=parts[0].get("text", "")
        )
                        
    # 串流的部分先維持原樣 (Tool calling 通常在 v0.1 先不做串流解析，太複雜了)
    def chat_completion_stream(
    self, 
    model: str, 
    messages: list[Message], 
    temperature: float = 0.7, 
    tools: list[dict] | None = None
    ) -> Generator[str, None, None]:
        url = f"{self.base_url}/{model}:streamGenerateContent?key={self.api_key}&alt=sse"
        payload = self._format_payload(messages, temperature, tools)
        with self._http_client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]
                    except json.JSONDecodeError:
                        continue