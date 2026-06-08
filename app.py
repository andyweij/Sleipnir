import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 引入你的 SDK 核心組件
from sleipnir_agent.agent import Agent
from sleipnir_agent.gemini_client import GeminiClient
from sleipnir_agent.openai_client import OpenAILikeClient
from tools.search import TavilySearchBackend, create_web_search_tool

load_dotenv()

app = FastAPI(title="Sleipnir Web API")

# 1. 定義前端傳過來的 API 請求結構 (Request Schema)
class ChatRequest(BaseModel):
    query: str
    use_web_search: bool = False
    # 新增參數：預設為雲端模型，前端可傳入 "local" 切換至地端
    backend_type: str = "cloud" 

# 系統啟動時，初始化共用的無狀態工具 (Search Backend)
search_backend = TavilySearchBackend(api_key=os.environ.get("TAVILY_API_KEY", ""))
web_search_tool = create_web_search_tool(search_backend)

# 2. 實作工廠函數：動態建立 Agent
def create_agent(backend_type: str) -> Agent:
    """根據前端請求，動態注入對應的 LLM Client"""
    
    if backend_type == "local":
        print("[System] 使用地端模型 (OpenAI Compatible)")
        # 這裡的 base_url 可以指向你公司內部的 vLLM 或 llama.cpp
        client = OpenAILikeClient(
            base_url=os.environ.get("LOCAL_LLM_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LOCAL_LLM_KEY", "EMPTY")
        )
        model_name = os.environ.get("LOCAL_MODEL_NAME", "llama-3-8b-instruct")
        
    else:
        print("[System] 使用雲端模型 (Gemini)")
        # 預設走雲端 Gemini API
        client = GeminiClient(
            api_key=os.environ.get("GEMINI_API_KEY", "")
        )
        model_name = os.environ.get("GEMINI_MODEL_NAME")
    # 組裝並回傳 Agent 實例
    return Agent(
        client=client,
        model=model_name,
        tools=[web_search_tool],
        max_iterations=5
    )

# 3. 實作 API 路由
@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 動態建立 Agent
        agent = create_agent(request.backend_type)
        final_query = request.query
        
        # 處理 UI 按下的「強制網頁搜尋」機制 (Pre-fetch Pattern)
        if request.use_web_search:
            print("[Backend] UI 觸發強制網頁搜尋")
            search_results = search_backend.search(request.query, max_results=3)
            
            context_str = "\n".join(
                [f"- [{res.title}]({res.url}): {res.content}" for res in search_results]
            )
            
            # Prompt Injection
            final_query = f"""
使用者詢問：{request.query}

系統已經自動在網路上為你搜尋了最新資訊，請「務必」參考以下搜尋結果來回答使用者的問題，並附上來源連結：
<search_results>
{context_str}
</search_results>
            """
            # 已經餵給它資料了，鎖死迭代次數防呆
            agent.max_iterations = 1 
            
        # 執行 Agent 推理
        response_text = agent.run(final_query)
        
        # 回傳給前端
        return {
            "answer": response_text,
            "backend_used": request.backend_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e