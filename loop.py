"""极简 Agent Loop：模型调用工具 -> 本地执行 -> 结果回传 -> 循环，直到模型不再调用工具。

运行前：pip install anthropic，并配置 ANTHROPIC_API_KEY（或 `ant auth login`）。
运行：python3 loop.py "北京和上海现在天气怎么样？"
"""

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5-5"

SYSTEM = "你是一个助手。需要实时信息时调用工具，不要编造结果。"


# ---------- 1. 工具的实现 ----------

def get_weather(city: str) -> str:
    """示例工具：真实项目里换成天气 API。"""
    fake = {"北京": "晴，22°C", "上海": "多云，26°C"}
    return f"{city}：{fake.get(city, '暂无数据')}"


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")


TOOL_IMPLS = {"get_weather": get_weather, "get_current_time": get_current_time}


# ---------- 2. 工具的 schema ----------

TOOLS = [
    {
        "name": "get_weather",
        "description": "查询某个城市的当前天气。",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_current_time",
        "description": "获取指定时区的当前时间。",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA 时区，如 Asia/Shanghai"}
            },
            "required": ["timezone"],
        },
    },
]


def call_tool(name: str, args: dict) -> tuple[str, bool]:
    """执行工具，返回 (结果文本, 是否出错)。出错也要回传，模型才能自行纠正。"""
    fn = TOOL_IMPLS.get(name)
    if fn is None:
        return f"未知工具：{name}", True
    try:
        return str(fn(**args)), False
    except Exception as e:  # noqa: BLE001 - 演示用，把任何异常交还给模型
        return f"{type(e).__name__}: {e}", True


# ---------- 3. 循环本体 ----------

def run_agent(user_input: str, max_turns: int = 10) -> None:
    messages = [{"role": "user", "content": user_input}]

    for _ in range(max_turns):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            betas=["server-side-fallback-2026-07-01"],  # 被安全策略拒绝时自动换模型重试
            fallbacks="default",
        )

        for block in response.content:
            if block.type == "text":
                print(block.text)

        if response.stop_reason == "end_turn":
            return

        # 服务端工具（如 web_search）跑太久会返回 pause_turn：原样带回即可续跑
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason != "tool_use":
            print(f"[stop_reason] {response.stop_reason}")
            return

        # 关键一步：assistant 的整条消息（含 tool_use 块）必须原样加入历史
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[tool] {block.name}({block.input})")
            result, is_error = call_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # 必须与 tool_use 的 id 对应
                "content": result,
                "is_error": is_error,
            })

        # 所有 tool_result 放在同一条 user 消息里一次回传
        messages.append({"role": "user", "content": tool_results})

    print(f"[!] 达到最大轮数 {max_turns}，停止")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "北京和上海现在天气怎么样？现在几点（Asia/Shanghai）？"
    run_agent(question)
    print("这是一个极简的示例agent loop")
