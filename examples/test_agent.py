import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from sleipnir_agent.agent import Agent
from sleipnir_agent.gemini_client import GeminiClient
from tools.registry import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a specific location."""
    # 在實際應用中這裡會去打氣象 API，這裡我們先用 mock 假資料
    if "Taipei" in location or "台北" in location:
        return f"Weather in {location} is 25 {unit}, slightly raining."
    return f"Weather in {location} is 20 {unit}, sunny."

@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers."""
    return a + b

def main():
    api_key = os.environ.get("GEMINI_API_KEY") 
    model_name = os.environ.get("GEMINI_MODEL_NAME")
    if not api_key:
        raise ValueError("請先設定 GEMINI_API_KEY 環境變數")
    if not model_name:
        raise ValueError("請先設定 GEMINI_MODEL_NAME 環境變數")

    client = GeminiClient(api_key=api_key)
    
    # 初始化 Agent
    agent = Agent(
        client=client,
        model=model_name,
        tools=[get_weather, calculate_sum],
        max_iterations=5
    )
    
    # 測試一個需要連續推理的問題：
    # LLM 應該要先找出台北的天氣，再算數學
    prompt = "台北現在天氣如何？順便幫我算一下 1234 加上 5678 是多少？"
    
    agent.run(prompt)

if __name__ == "__main__":
    main()