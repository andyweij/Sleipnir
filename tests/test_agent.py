from unittest.mock import MagicMock

from sleipnir_agent.agent import Agent
from sleipnir_agent.client import Message, ToolCall
from tools.registry import tool


# 準備一個測試用的 Dummy Tool
@tool
def dummy_weather(location: str) -> str:
    """Get weather."""
    return f"Weather in {location} is sunny."

class TestAgent:
    def test_agent_early_stopping(self):
        """測試情境 1：Agent 不需要呼叫工具，直接給出答案"""
        
        # 建立一個假的 Client
        mock_client = MagicMock()
        
        # 設定這個假 Client 呼叫 chat_completion 時要回傳什麼
        mock_client.chat_completion.return_value = Message(
            role="assistant", 
            content="I am a mocked response."
        )
        
        agent = Agent(client=mock_client, model="dummy-model")
        result = agent.run("Hello, who are you?")
        
        # 驗證結果
        assert result == "I am a mocked response."
        # 驗證 Client 確實只被呼叫了一次
        assert mock_client.chat_completion.call_count == 1

    def test_agent_tool_calling_loop(self):
        """測試情境 2：Agent 決定呼叫工具，然後再給出最終答案"""
        
        mock_client = MagicMock()
        
        # 【Pythonic 慣用法】: side_effect 可以傳入一個列表
        # 代表第一次呼叫回傳列表的第 1 個值，第二次呼叫回傳第 2 個值
        mock_client.chat_completion.side_effect = [
            # 第一次：LLM 決定呼叫 dummy_weather 工具
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="call_123", name="dummy_weather", arguments='{"location": "Taipei"}')
                ]
            ),
            # 第二次：LLM 看到工具回傳的 Observation 後，給出最終答案
            Message(
                role="assistant",
                content="The weather in Taipei is sunny."
            )
        ]
        
        agent = Agent(client=mock_client, model="dummy-model", tools=[dummy_weather])
        result = agent.run("What is the weather in Taipei?")
        
        assert result == "The weather in Taipei is sunny."
        # 驗證 ReAct Loop 確實跑了兩圈（思考+呼叫工具 -> 總結）
        assert mock_client.chat_completion.call_count == 2
    def test_agent_max_iterations_reached(self):
        """測試情境 3：Agent 陷入無窮迴圈，觸發 max_iterations 安全機制"""
        mock_client = MagicMock()
        
        # 讓 LLM 一直鬼打牆呼叫同一個工具
        mock_client.chat_completion.return_value = Message(
            role="assistant",
            tool_calls=[ToolCall(id="call_999", name="dummy_weather", arguments='{"location": "Taipei"}')]
        )
        
        # 刻意把 max_iterations 設很小，加速測試
        agent = Agent(client=mock_client, model="dummy-model", tools=[dummy_weather], max_iterations=2)
        result = agent.run("What is the weather?")
        
        # 驗證是否被系統強制中斷
        assert "Reached maximum iterations" in result
        assert mock_client.chat_completion.call_count == 2

    def test_agent_tool_execution_error(self):
        """測試情境 4：LLM 傳了壞掉的 JSON 參數，或是呼叫不存在的工具"""
        mock_client = MagicMock()
        
        mock_client.chat_completion.side_effect = [
            # 第一次：LLM 給了不合法的 JSON 字串
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="call_err", name="dummy_weather", arguments='{Bad JSON]')]
            ),
            # 第二次：Agent 應該要把 Error 訊息回傳給 LLM，然後 LLM 道歉並給出回答
            Message(
                role="assistant",
                content="Sorry, I messed up the JSON."
            )
        ]
        
        agent = Agent(client=mock_client, model="dummy-model", tools=[dummy_weather])
        result = agent.run("Test broken JSON")
        
        assert result == "Sorry, I messed up the JSON."