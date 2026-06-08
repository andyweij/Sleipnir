import os
import sys
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
    if not model_name:
        raise ValueError("請先設定 GEMINI_MODEL_NAME 環境變數")
    # 1. 實例化 Client 與 Backend
    client = GeminiClient(api_key=gemini_key)
    search_backend = TavilySearchBackend(api_key=tavily_key)

    # 2. 透過工廠函數建立包含 backend 依賴的工具
    web_search_tool = create_web_search_tool(search_backend)

    # 3. 初始化 Agent
    agent = Agent(client=client, model=model_name, tools=[web_search_tool], max_iterations=5)

    # 測試一個需要即時搜尋的問題
    prompt = "請查一下今天台灣的最新新聞重點，並附上來源網址。"

    agent.run(prompt)


if __name__ == "__main__":
    main()
