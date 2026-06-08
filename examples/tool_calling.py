import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from sleipnir_agent.client import Message
from sleipnir_agent.gemini_client import GeminiClient
from tools.registry import tool


# 1. 定義我們的工具
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a specific location."""
    return f"Weather in {location} is 25 {unit}."


def main():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL_NAME")
    if not gemini_key:
        raise ValueError("請先設定 GEMINI_API_KEY 環境變數")
    if not model_name:
        raise ValueError("請先設定 GEMINI_MODEL_NAME 環境變數")
    client = GeminiClient(api_key=gemini_key)

    # 2. 故意問一個需要呼叫工具的問題
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="請問台北現在的天氣如何？"),
    ]

    # 3. 將 @tool 自動生成的 OpenAI Schema 餵給 Client
    tools_schema = [get_weather.to_openai_schema()]

    print("=== 發送請求給 LLM ===")
    response_msg = client.chat_completion(model=model_name, messages=messages, tools=tools_schema)

    # 4. 驗證 LLM 是否正確判斷需要呼叫工具
    if response_msg.tool_calls:
        print("\n✅ 成功！LLM 決定呼叫工具：")
        tool_call = response_msg.tool_calls[0]
        print(f"Tool Name: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")
    else:
        print("\n❌ LLM 沒有呼叫工具，直接回答了：")
        print(response_msg.content)


if __name__ == "__main__":
    main()
