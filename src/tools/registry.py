import inspect
from typing import Any, Callable

class Tool:
    """
    將普通的 Python function 包裝成 LLM 可以理解的 Tool 結構。
    (概念上等同於 Java 中利用 Reflection 讀取 Annotation 並封裝的 Wrapper 類別)
    """
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        # Pythonic: __doc__ 可以取得函式的 docstring。若無則給空字串防呆
        self.description = inspect.cleandoc(func.__doc__ or "")
        self.parameters = self._generate_schema(func)

    def _generate_schema(self, func: Callable) -> dict:
        """利用 inspect 模組，從 Type Hints 動態生成 JSON Schema"""
        sig = inspect.signature(func)
        properties = {}
        required = []

        # v0.1 簡化版型別映射
        type_mapping = {
            int: "integer",
            float: "number",
            bool: "boolean",
            str: "string",
        }

        for name, param in sig.parameters.items():
            # 取得參數的 Type Hint，如果沒寫預設為 string
            python_type = param.annotation
            json_type = type_mapping.get(python_type, "string")

            properties[name] = {
                "type": json_type,
                "description": f"Parameter: {name}" 
            }
            
            # Pythonic: 判斷參數是否有預設值。如果沒有預設值，就是必填項 (required)
            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    # Pythonic: 實作 Dunder Method __call__
    # 這讓 Tool 的實例可以像普通 function 一樣被呼叫，例如 my_tool(a=1)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def to_openai_schema(self) -> dict:
        """輸出標準的 OpenAI Function Calling Schema 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

def tool(func: Callable) -> Tool:
    """
    這是一個 Decorator 函式。
    用法:
        @tool
        def my_function(x: int): ...
    """
    return Tool(func)