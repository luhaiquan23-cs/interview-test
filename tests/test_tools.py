"""
工具系统单元测试 (Unit Tests for Tools)
----------------------------------------------------------------------
覆盖范围:
  1. SafeCalculator: 基础运算能力验证与 AST 安全沙箱防护（阻断任意代码执行 RCE）；
  2. CalculatorTool: 工具封装与执行验证；
  3. SearchTool: 模拟知识库检索匹配验证；
  4. TodoTool / TodoStore: 待办事项全生命周期 CRUD（增删查改）与按 Session 物理隔离；
  5. WeatherTool: 城市天气查询功能验证；
  6. ToolRegistry: 工具注册、Schema 导出与错误自纠错返回验证。
"""

import pytest
from agent.tools.builtin import (
    SafeCalculator,
    CalculatorTool,
    SearchTool,
    TodoStore,
    TodoTool,
    WeatherTool,
)
from agent.tools.base import ToolRegistry, tool
from agent.core.schema import ToolCall


def test_safe_calculator_basic():
    """测试安全计算器的四则运算、幂运算与内置数学函数"""
    assert SafeCalculator.evaluate("1 + 1") == 2
    assert SafeCalculator.evaluate("10 * (5 - 2)") == 30
    assert SafeCalculator.evaluate("2 ** 8") == 256
    assert SafeCalculator.evaluate("sqrt(144)") == 12.0
    assert SafeCalculator.evaluate("round(10 / 3, 2)") == 3.33


def test_safe_calculator_security():
    """
    安全沙箱测试: 验证禁止一切任意系统代码注入（RCE 漏洞防护）。
    尝试导入 os 或读取本地文件均会被 AST 语法节点解析拦截并抛出异常。
    """
    with pytest.raises(Exception):
        SafeCalculator.evaluate("__import__('os').system('ls')")
    with pytest.raises(Exception):
        SafeCalculator.evaluate("open('/etc/passwd')")


def test_calculator_tool():
    """测试 CalculatorTool 工具对象入参与返回值格式"""
    calc = CalculatorTool()
    res = calc.run(expression="15 * 12")
    assert res["result"] == 180


def test_search_tool():
    """测试 SearchTool 模拟知识检索功能"""
    search = SearchTool()
    res = search.run(query="deepseek agent")
    assert "results" in res
    assert len(res["results"]) > 0


def test_todo_tool_crud():
    """测试 TodoTool 完整的增删查改（CRUD）闭环流程"""
    store = TodoStore()
    todo_tool = TodoTool(store=store)

    # 1. 增 (Add)
    res_add = todo_tool.run(action="add", title="Write weekly report", priority="high", session_id="s1")
    assert res_add["status"] == "success"
    todo_id = res_add["todo"]["id"]

    # 2. 查 (List)
    res_list = todo_tool.run(action="list", session_id="s1")
    assert res_list["total"] == 1
    assert res_list["todos"][0]["title"] == "Write weekly report"

    # 3. 改 (Complete 标记完成)
    res_comp = todo_tool.run(action="complete", todo_id=todo_id, session_id="s1")
    assert res_comp["status"] == "success"
    assert res_comp["todo"]["completed"] is True

    # 4. 删 (Delete)
    res_del = todo_tool.run(action="delete", todo_id=todo_id, session_id="s1")
    assert res_del["status"] == "success"
    assert len(store.list("s1")) == 0


def test_todo_session_isolation():
    """
    测试 TodoTool 核心特性: 按 Session 独立物理分桶隔离。
    窗口 1 增加的待办与窗口 2 增加的待办互不干扰、互不可见。
    """
    store = TodoStore()
    todo_tool = TodoTool(store=store)

    # Session 1 新增待办
    todo_tool.run(action="add", title="Check weather and pack bag", session_id="session_1")

    # Session 2 新增完全不同的待办
    todo_tool.run(action="add", title="Submit Q3 revenue summary", session_id="session_2")

    # 验证 Session 1 仅包含其自身的数据
    s1_todos = todo_tool.run(action="list", session_id="session_1")["todos"]
    s2_todos = todo_tool.run(action="list", session_id="session_2")["todos"]

    assert len(s1_todos) == 1
    assert "weather" in s1_todos[0]["title"]

    assert len(s2_todos) == 1
    assert "revenue" in s2_todos[0]["title"]


def test_weather_tool():
    """测试 WeatherTool 城市天气查询功能"""
    weather = WeatherTool()
    res = weather.run(city="Xiamen")
    assert res["city"] == "Xiamen"
    assert "temperature" in res


def test_tool_registry():
    """测试 ToolRegistry 工具动态注册、Schema 输出与未注册工具的异常自愈捕获"""
    registry = ToolRegistry()
    calc = CalculatorTool()
    registry.register(calc)

    # 验证导出的 JSON Schema 规范
    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "calculator"

    # 正常调用分发
    tc = ToolCall(id="tc_1", name="calculator", arguments={"expression": "100 / 4"})
    res = registry.execute(tc)
    assert res.success is True
    assert "25" in res.content

    # 异常调用分发（请求未注册的工具）: 不抛出系统异常，包装为友好错误提示供大模型纠错
    tc_invalid = ToolCall(id="tc_2", name="non_existent_tool", arguments={})
    res_err = registry.execute(tc_invalid)
    assert res_err.success is False
    assert "Error: Tool 'non_existent_tool' is not registered" in res_err.content
