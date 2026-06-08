import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from tools.registry import tool


# 1. 測試定義工具：這個 docstring 和 type hints 會被自動解析
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a specific location.
    """
    return f"Weather in {location} is 25 {unit}."


@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers."""
    return a + b


def main():
    print("=== 測試天氣工具 Schema ===")
    print(f"Tool Name: {get_weather.name}")
    print(json.dumps(get_weather.to_openai_schema(), indent=2, ensure_ascii=False))

    print("\n=== 測試執行工具 ===")
    # 確保原本的 function 還是能被正常執行 (歸功於 __call__)
    result = calculate_sum(a=5, b=10)
    print(f"calculate_sum(5, 10) = {result}")


if __name__ == "__main__":
    main()
