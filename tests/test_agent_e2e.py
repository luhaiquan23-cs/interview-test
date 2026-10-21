"""
端到端真实 API 集成测试 (End-to-End Integration Tests with DeepSeek)
----------------------------------------------------------------------
通过真实网络调用 DeepSeek API（deepseek-chat 模型）全面验证全链路功能：
  1. 纯对话测试 (Direct Conversation): 模型直接回答，无冗余工具调用；
  2. 计算器工具真实调用 (Calculator Tool Calling): 模型自主决策调用 calculator 并计算 384*25=9600；
  3. 待办事项与多轮追问 (Todo CRUD & Follow-up): 新增待办 -> 第二轮追问待办清单；
  4. 多窗口会话与待办存储隔离 (Multi-Session Isolation): 
     - 窗口 1: 查厦门天气 + 记买动车票待办
     - 窗口 2: 周报撰写要点 + 记周报待办
     - 物理核验 Todo 存储分桶，验证两窗口完全隔离。
"""

import pytest
import os
from agent.core.runtime import AgentRuntime
from agent.tools.builtin import TodoTool

# 若未配置真实 DEEPSEEK_API_KEY，自动跳过 E2E 测试，防止 CI/CD 本地运行报错
pytestmark = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not configured in .env",
)


def test_e2e_direct_conversation():
    """
    E2E 测试 1: 纯对话交互（不需要工具调用）
    验证:
      1. 模型能够直接生成自然语言回答；
      2. 终态为 direct_response；
      3. 不触发任何无关的工具调用。
    """
    runtime = AgentRuntime()
    answer, trace = runtime.run("你好，请用一句话简短介绍你自己", session_id="e2e_direct")

    assert trace.success is True
    assert len(answer) > 0
    assert len(trace.steps) >= 1
    assert trace.steps[-1].action == "direct_response"


def test_e2e_tool_calculator():
    """
    E2E 测试 2: 单次工具调用与精确计算
    验证:
      1. 模型准确识别数学计算需求，自主决策生成 calculator 工具调用；
      2. 成功执行 384 * 25 计算；
      3. 最终回复准确包含计算结果 9600。
    """
    runtime = AgentRuntime()
    answer, trace = runtime.run("请帮我计算 384 * 25 等于多少？必须调用计算器工具计算。", session_id="e2e_calc")

    assert trace.success is True
    assert "9600" in answer

    # 核验执行链路中确实调用了 calculator 工具
    tool_calls_made = [
        tc.name
        for step in trace.steps
        for tc in step.tool_calls
    ]
    assert "calculator" in tool_calls_made


def test_e2e_todo_and_followup():
    """
    E2E 测试 3: 待办工具调用与多轮追问 (Follow-up)
    步骤:
      - 轮次 1: 用户要求新增“周五下午准备述职汇报PPT”待办 -> 模型调用 todo(action='add')
      - 轮次 2: 用户紧接着追问“我刚才添加了什么待办？” -> 模型基于上下文调用 todo(action='list') 并准确回答
    """
    runtime = AgentRuntime()
    session_id = "e2e_todo_session"

    # 第一轮: 新增待办
    ans1, trace1 = runtime.run(
        "帮我添加一个待办事项：周五下午准备述职汇报PPT",
        session_id=session_id,
    )
    assert trace1.success is True
    tool_calls_1 = [tc.name for step in trace1.steps for tc in step.tool_calls]
    assert "todo" in tool_calls_1

    # 第二轮: 多轮追问
    ans2, trace2 = runtime.run(
        "我刚才添加了什么待办？帮我查看一下所有待办清单",
        session_id=session_id,
    )
    assert trace2.success is True
    # 验证模型回答中包含上一轮添加的待办关键词
    assert "述职" in ans2 or "PPT" in ans2


def test_e2e_multi_session_isolation():
    """
    E2E 测试 4: 多窗口并行与数据隔离测试
    模拟真实用户在同一客户端开启两个窗口：
      - 窗口 1 (window_1): 查厦门天气并记录动车票待办
      - 窗口 2 (window_2): 询问周报建议并记录周报待办
    验证:
      - 底层 TodoStore 物理存储中，window_1 的列表仅包含厦门动车票，不含周报；
      - window_2 的列表仅包含周报，不含厦门动车票；
      - 完美实现会话隔离。
    """
    runtime = AgentRuntime()

    # 窗口 1 操作
    ans1, trace1 = runtime.run(
        "查一下厦门天气，然后加一个待办：买去厦门的动车票",
        session_id="window_1",
    )
    assert trace1.success is True

    # 窗口 2 操作
    ans2, trace2 = runtime.run(
        "写周报的核心结构是什么？顺便加一个待办：周五提交周报",
        session_id="window_2",
    )
    assert trace2.success is True

    # 获取底层 Todo 工具实例并校验内部存储状态
    todo_tool = runtime.tools.get_tool("todo")
    assert isinstance(todo_tool, TodoTool)

    w1_todos = todo_tool.store.list("window_1")
    w2_todos = todo_tool.store.list("window_2")

    # 校验窗口 1 数据纯净性
    assert len(w1_todos) >= 1
    assert any("厦门" in item["title"] or "动车" in item["title"] for item in w1_todos)
    assert not any("周报" in item["title"] for item in w1_todos)

    # 校验窗口 2 数据纯净性
    assert len(w2_todos) >= 1
    assert any("周报" in item["title"] for item in w2_todos)
    assert not any("厦门" in item["title"] for item in w2_todos)
