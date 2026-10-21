"""
Runtime 核心循环单元测试 (Unit Tests for Agent Runtime Loop)
----------------------------------------------------------------------
通过确定性的 Mock LLM 客户端精确测试 4-Step 执行循环的状态机跃迁：
  - Step 1: 接收输入
  - Step 2: 判断是直接回复 (direct_response) 还是调用工具 (tool_call)
  - Step 3: 安全调用工具并封装 ToolResult
  - Step 4: 根据工具返回结果判断继续循环还是向用户输出最终答案
  
包含边界与防护用例:
  - 最大推理步数防死循环熔断保护 (Max Steps Guard);
  - 工具不存在时的错误捕获与自愈机制。
"""

import pytest
from typing import List, Dict, Any, Optional, Tuple
from agent.core.runtime import AgentRuntime
from agent.core.schema import Message, ToolCall
from agent.tools.base import ToolRegistry
from agent.tools.builtin import CalculatorTool, TodoTool, TodoStore


class MockLLMClient:
    """
    可预测的 Mock LLM 客户端。
    用于按预设顺序逐步返回特定响应（包括思考过程、工具调用与文本回复），无需消耗真实 API。
    """

    def __init__(self, canned_responses: List[Tuple[Optional[str], Optional[str], List[ToolCall], str]]):
        self.canned_responses = list(canned_responses)
        self.call_history: List[List[Message]] = []

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.6,
    ) -> Tuple[Optional[str], Optional[str], List[ToolCall], str]:
        self.call_history.append(messages)
        if not self.canned_responses:
            return "Fallback reply", None, [], "stop"
        return self.canned_responses.pop(0)


def test_runtime_step2_direct_reply():
    """
    测试 Step 2 分支 A: 直接回复（无需工具调用）
    验证: 当模型未返回 tool_calls 时，Runtime 立即将思考过程与回复写入上下文并退出循环。
    """
    mock_llm = MockLLMClient([
        ("Hello! How can I assist you?", "Thinking: no tool needed.", [], "stop")
    ])
    runtime = AgentRuntime(llm_client=mock_llm)

    answer, trace = runtime.run("Hi", session_id="test_direct")

    assert answer == "Hello! How can I assist you?"
    assert len(trace.steps) == 1
    assert trace.steps[0].action == "direct_response"
    assert trace.steps[0].thought == "Thinking: no tool needed."
    assert trace.success is True


def test_runtime_step3_and_4_tool_call_loop():
    """
    测试 Step 3 & Step 4: 工具调用与状态闭环
    模拟场景:
      1. 模型首轮判断需要调用 calculator 工具 (Step 2 -> Step 3)
      2. 工具执行计算并将结果追加至上下文
      3. 模型在第二轮观察到计算结果，输出最终解答，结束循环 (Step 4 -> Step 2 -> direct_response)
    """
    tool_call = ToolCall(id="call_123", name="calculator", arguments={"expression": "25 * 4"})

    mock_llm = MockLLMClient([
        # 第一轮: 模型提出工具调用请求
        (None, "Thinking: compute 25 * 4 using calculator", [tool_call], "tool_calls"),
        # 第二轮: 模型根据工具返回的 100 进行总结回复
        ("The result of 25 * 4 is 100.", "Thinking: calculation done, answer user.", [], "stop"),
    ])

    runtime = AgentRuntime(llm_client=mock_llm)
    answer, trace = runtime.run("What is 25 * 4?", session_id="test_calc")

    assert "100" in answer
    assert len(trace.steps) == 2

    # 验证步骤 1 为工具执行
    assert trace.steps[0].action == "tool_call"
    assert trace.steps[0].tool_calls[0].name == "calculator"
    assert "100" in trace.steps[0].tool_results[0].content

    # 验证步骤 2 为最终回复
    assert trace.steps[1].action == "direct_response"
    assert trace.steps[1].response_content == "The result of 25 * 4 is 100."


def test_runtime_max_steps_guard():
    """
    测试最大步数熔断保护:
    模拟模型陷入死循环无限发起工具调用。验证 Runtime 在达到 max_steps_per_turn 时安全截断并输出提示，杜绝死循环。
    """
    infinite_calls = [
        (None, "looping", [ToolCall(id=f"c_{i}", name="calculator", arguments={"expression": "1+1"})], "tool_calls")
        for i in range(10)
    ]
    mock_llm = MockLLMClient(infinite_calls)

    # 限制单轮最多允许 3 步迭代
    runtime = AgentRuntime(llm_client=mock_llm, max_steps_per_turn=3)
    answer, trace = runtime.run("Infinite loop test", session_id="test_infinite")

    # 验证安全停止在第 3 步
    assert len(trace.steps) == 3
    assert "maximum reasoning steps limit" in answer


def test_runtime_error_handling_invalid_tool():
    """
    测试异常容错与自愈机制:
    模拟模型尝试调用一个不存在的工具 'magic_tool'。
    Runtime 捕获未注册错误，作为 Observation 传入上下文，模型感知错误后友好降级回复。
    """
    mock_llm = MockLLMClient([
        (None, "call unknown", [ToolCall(id="err_1", name="magic_tool", arguments={})], "tool_calls"),
        ("I apologize, magic_tool is not available.", "Handle failure", [], "stop")
    ])

    runtime = AgentRuntime(llm_client=mock_llm)
    answer, trace = runtime.run("Do magic", session_id="test_err")

    assert len(trace.steps) == 2
    assert trace.steps[0].tool_results[0].success is False
    assert "not registered" in trace.steps[0].tool_results[0].content
    assert "magic_tool is not available" in answer
