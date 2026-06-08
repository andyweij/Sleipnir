from typing import Protocol

import httpx
from pydantic import BaseModel

from .registry import tool


# 1. 定義回傳的資料結構 (DTO)
class SearchResult(BaseModel):
    title: str
    url: str
    content: str

# 2. 定義 SearchBackend 介面 (Protocol)
class SearchBackend(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...

# 3. 實作 Tavily Backend
class TavilySearchBackend:
    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"
        self._http_client = httpx.Client(timeout=timeout)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": max_results
        }
        
        response = self._http_client.post(self.base_url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", "")
            ))
            
        return results

# 4. 建立一個工廠函數來生成 @tool，這樣我們才能把 Backend 注入進去
def create_web_search_tool(backend: SearchBackend):
    """
    這是一個閉包 (Closure) 工廠，用來將 SearchBackend 實例注入到 tool 中。
    這樣 Agent 就能直接呼叫這個回傳的 function。
    """
    @tool
    def web_search(query: str) -> str:
        """
        Search the web for current and up-to-date information.
        Use this tool when you need facts, news, or knowledge you don't already know.
        """
        results = backend.search(query, max_results=3)
        if not results:
            return "No relevant results found on the web."
        
        # 根據 Roadmap：來源引用(每個結論帶 URL)
        formatted_results = []
        for i, res in enumerate(results, 1):
            formatted_results.append(f"[{i}] {res.title}\nURL: {res.url}\nSummary: {res.content}\n")
            
        return "\n".join(formatted_results)
        
    return web_search