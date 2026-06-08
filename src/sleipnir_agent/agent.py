import datetime
import json
from typing import Any

from tools.registry import Tool

from .client import LLMClient, Message
from .events import TraceEvent

DEFAULT_SYSTEM_PROMPT = """You are an advanced reasoning agent.
You have access to specific tools. Always use them to gather factual information.

Core Directives:
1. Self-Evaluation: Before giving a final answer, evaluate if you have enough information. If not, use the search tool again with a different query.
2. Multi-Source Verification: Never rely on a single source for sensitive or factual questions. Cross-reference at least two different sources before answering.
3. Citation: When providing the final answer, always cite your sources using the URLs provided by the search tool.
4. Knowledge Verification (Crucial): You may use your internal knowledge to form an initial hypothesis, but you MUST NEVER output a final answer based solely on it. You MUST call the `web_search` tool to cross-check and verify your internal knowledge against current web sources before answering.
"""

class Agent:
    def __init__(self, client: LLMClient, 
                 model: str, 
                 tools: list[Tool] | None = None, 
                 max_iterations: int = 10,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT
                 ):
        self.client = client
        self.model = model
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        # Pythonic: 建立一個字典 (Hash Map)，方便等一下用工具名稱以 O(1) 複雜度快速尋找並執行
        self._tool_map = {tool.name: tool for tool in self.tools}
        
        # 預先將所有工具轉成 OpenAI 要求的 JSON Schema
        self._tool_schemas = [tool.to_openai_schema() for tool in self.tools]
        self.trace: list[TraceEvent] = []  # 加這個

    def run(self, user_input: str) -> str:
        # 【Pythonic 慣用法】: 取得當前時間並格式化
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
        
        # 動態將環境變數 (時間) 注入到系統提示詞的最底部
        dynamic_system_prompt = f"{self.system_prompt}\n\n[System Info]\nCurrent Time: {current_time}"

        """執行 ReAct Loop 直到得出最終答案或達到次數上限"""
        
        # 1. 初始化 Short-term working memory
        messages = [
            Message(role="system", content=dynamic_system_prompt),
            Message(role="user", content=user_input)
        ]

        print(f"\n[User]: {user_input}")

        # 2. 進入 Loop Controller
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # 呼叫 LLM
            response_msg = self.client.chat_completion(
                model=self.model,
                messages=messages,
                tools=self._tool_schemas if self._tool_schemas else None
            )
            self.trace.append(TraceEvent(
                kind="llm_response",
                data={"has_tool_calls": bool(response_msg.tool_calls)}
            ))
            # 將 LLM 的回覆 (可能是文字，也可能是 tool_calls) 加入對話紀錄
            messages.append(response_msg)

            # 判斷是否需要呼叫工具
            if response_msg.tool_calls:
                for tool_call in response_msg.tool_calls:
                    self.trace.append(TraceEvent(
                        kind="tool_call",
                        data={"name": tool_call.name, "arguments": tool_call.arguments}
                    ))
                    
                    tool_result = self._execute_tool(tool_call.name, tool_call.arguments)
                    
                    self.trace.append(TraceEvent(
                        kind="tool_result",
                        data={"name": tool_call.name, "result": str(tool_result)[:500]}
                    ))
                    print(f"[Action]: Calling tool '{tool_call.name}' with args {tool_call.arguments}")
                    
                    # 執行工具
                    tool_result = self._execute_tool(tool_call.name, tool_call.arguments)
                    print(f"[Observation]: {tool_result}")
                    
                    # 將工具執行的結果放回 memory
                    # 注意：role 必須是 tool，且必須帶上對應的 id 與 name
                    messages.append(Message(
                        role="tool",
                        content=str(tool_result),
                        name=tool_call.name,
                        tool_call_id=tool_call.id
                    ))
                
                # 迴圈繼續，讓 LLM 看到 Observation 後再做決定
                continue
            else:
                self.trace.append(TraceEvent(
                    kind="final_answer",
                    data={"content": response_msg.content}
                ))
                return response_msg.content or ""
            # 如果沒有 tool_calls，代表 LLM 認為資訊充足，已經給出文字解答 (Early Stopping)
            print(f"\n[Final Answer]: {response_msg.content}")
            return response_msg.content or ""

        # 迴圈跑完還是沒得到答案，觸發安全機制
        return "Error: Reached maximum iterations. Agent stopped."
        
    def _execute_tool(self, name: str, arguments: str) -> Any:
        """解析 JSON 參數並執行對應的 Python function"""
        if name not in self._tool_map:
            return f"Error: Tool '{name}' not found."
        
        try:
            # 將 LLM 給的 JSON 字串反序列化為 Python 字典
            kwargs = json.loads(arguments)
            tool = self._tool_map[name]
            
            # Pythonic 慣用法: Dictionary Unpacking (**kwargs)
            # 等同於自動把字典裡的 key-value 對應到 function 的參數上
            return tool(**kwargs) 
        except json.JSONDecodeError:
            return "Error: Invalid JSON arguments."
        except Exception as e:
            return f"Error executing tool: {str(e)}"