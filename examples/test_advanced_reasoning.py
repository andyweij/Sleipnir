import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent / "src"))

from sleipnir_agent.agent import Agent
from sleipnir_agent.gemini_client import GeminiClient
from tools.search import TavilySearchBackend, create_web_search_tool

load_dotenv()

def main():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL_NAME")
    if not gemini_key or not tavily_key:
        raise ValueError("請確認 .env 檔案中已設定 GEMINI_API_KEY 與 TAVILY_API_KEY")

    client = GeminiClient(api_key=gemini_key)
    search_backend = TavilySearchBackend(api_key=tavily_key)
    web_search_tool = create_web_search_tool(search_backend)
    
    # 這裡可以把 max_iterations 調高一點，因為多源比對需要更多次的 Action
    agent = Agent(
        client=client,
        model=model_name, # 如果你有 pro 權限，用 gemini-1.5-pro 展現的推理能力會更強
        tools=[web_search_tool],
        max_iterations=10 
    )
    
    # 測試多源比對的複雜 Query
    # 這個問題模型不可能靠單一搜尋字串一次找齊，它必須先搜 Google I/O，再搜 Apple WWDC，然後自我評估整合
    prompt = "請幫我比較 2024 年 Google I/O 與 Apple WWDC 兩場大會上，針對『手機端側 AI (On-device AI)』的發展重點有何不同？請務必列出你的參考來源。"
    
    print("=== 開始執行進階推理測試 ===")
    agent.run(prompt)

if __name__ == "__main__":
    main()