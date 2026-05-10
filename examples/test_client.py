import sys
import os
# 確保能 import 到 src 下的 sleipnir
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sleipnir_agent.client import Message
from sleipnir_agent.gemini_client import GeminiClient
from sleipnir_agent.openai_client import OpenAILikeClient

def main():
    # 請把這裡換成你 Java 平台的實際 URL 與 Model 名稱
    gemini_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL_NAME")
    if not gemini_key:
        raise ValueError("請先設定 GEMINI_API_KEY 環境變數")
    if not model_name:
        raise ValueError("請先設定 GEMINI_MODEL_NAME 環境變數")
    client = GeminiClient(api_key=gemini_key)
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="請用繁體中文解釋什麼是『八腳神馬 Sleipnir』，限制在 50 字以內。")
    ]

    print("=== 測試非串流 (Non-streaming) ===")
    response = client.chat_completion(model=model_name, messages=messages)
    print(response)
    
    print("\n=== 測試串流 (Streaming) ===")
    for token in client.chat_completion_stream(model=model_name, messages=messages):
        # end="" 讓 print 不換行，flush=True 強制立刻輸出到終端機
        print(token, end="", flush=True)
    print()

if __name__ == "__main__":
    main()