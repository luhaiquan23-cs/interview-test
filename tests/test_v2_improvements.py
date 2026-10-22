"""
V2 架构重构专项测试套件 (Targeted Tests for Reviewer Feedback)
----------------------------------------------------------------------
针对面试官反馈的 5 大痛点逐一构建的严密测试用例：
  1. test_atomic_turn_compaction_never_breaks_tool_calls:
     验证原子轮次切分机制，彻底杜绝切散 assistant(tool_calls) 与 tool(result) 的问题。
  2. test_rolling_summary_bounded_growth:
     验证滚动定长摘要机制，连续交互数十轮后摘要长度依然严格受限于固定预算，解决“只增不减”。
  3. test_sqlite_persistence_across_process_restart:
     验证会话、上下文与待办事项的 SQLite 真实落盘，模拟进程退出并重启后的无损恢复。
  4. test_cascade_delete_cleans_todos_and_context:
     验证会话删除时的级联清理机制，删掉 session 时同步物理清空该窗口的 Todo 待办事项。
  5. test_session_state_machine_transition:
     验证 SessionState 深度参与运行时状态机驱动 (idle -> running -> idle, 轮次与步数自增)。
"""

import pytest
import os
import gc
from agent.core.schema import Message, ToolCall, SessionState
from agent.core.context import ContextManager
from agent.core.session import SessionManager
from agent.core.runtime import AgentRuntime
from agent.storage.db import SQLiteStorage
from agent.tools.builtin import TodoStore, TodoTool


def test_atomic_turn_compaction_never_breaks_tool_calls():
    """
    痛点 1 验证: “压缩按条数切开，会拆散 assistant 的 tool call 和后面的 tool 结果”
    
    测试逻辑:
      构造 3 轮完整交互，其中第 2 轮包含复合工具调用:
        Turn 2: [User] -> [Assistant(tool_calls)] -> [Tool(obs)] -> [Assistant(final)]
      设置 max_turns=2, keep_recent_turns=1 触发压缩。
      断言:
        保留的消息中，任何 role='tool' 的消息前面必须紧跟包含该 tool_call_id 的 assistant 消息，
        或者整个 Turn 2 一并被压缩进摘要，绝不会被从中间切断！
    """
    ctx = ContextManager(max_turns=2, keep_recent_turns=1)

    # Turn 1
    ctx.add_user_message("Turn 1 question")
    ctx.add_assistant_message(content="Turn 1 answer")

    # Turn 2 (带工具调用)
    ctx.add_user_message("Turn 2 question")
    tc_dict = [{"id": "call_999", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}]
    ctx.add_assistant_message(content="Thinking...", tool_calls=tc_dict)
    ctx.add_tool_message(tool_call_id="call_999", name="calculator", content="Result: 42")
    ctx.add_assistant_message(content="The final result is 42.")

    # Turn 3 (触发压缩)
    ctx.add_user_message("Turn 3 question")
    ctx.add_assistant_message(content="Turn 3 answer")

    messages = ctx.get_messages()

    # 检查所有现存消息的原子配对完整性
    for i, m in enumerate(messages):
        if m.role == "tool":
            prev_m = messages[i - 1]
            # 工具返回的前一条消息必须是 assistant 且必须包含对应的 tool_call_id
            assert prev_m.role == "assistant"
            assert prev_m.tool_calls is not None
            assert any(tc["id"] == m.tool_call_id for tc in prev_m.tool_calls)


def test_rolling_summary_bounded_growth():
    """
    痛点 2 验证: “摘要只增不减”
    
    测试逻辑:
      连续触发 15 轮对话压缩，设置 max_summary_chars=300。
      断言:
        即使经历多次增量压缩，最终的 _summary 长度依然被严格限制在 300 字符内，杜绝无限膨胀。
    """
    ctx = ContextManager(max_turns=2, keep_recent_turns=1, max_summary_chars=300)

    for i in range(15):
        ctx.add_user_message(f"This is a relatively long user prompt for turn {i} with lots of context words.")
        ctx.add_assistant_message(content=f"This is a detailed response for turn {i} providing valuable information.")

    # 验证摘要长度在预算限制内
    assert ctx._summary is not None
    assert len(ctx._summary) <= 300


def test_sqlite_persistence_across_process_restart():
    """
    痛点 3 验证: “数据在内存字典里，进程退出即消失”
    
    测试逻辑:
      1. 使用独立 SQLite 数据库文件；
      2. 启动 Process 1: 写入窗口 'window_work' 的会话、对话历史和待办事项；
      3. 模拟进程完全退出（显式关闭连接）；
      4. 启动 Process 2: 重新从该 SQLite 文件加载 Runtime；
      5. 断言: 窗口 'window_work' 的状态机、上下文消息与待办事项完好无损地恢复！
    """
    test_db_path = "data/test_persist.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    try:
        # ---- 进程 1 ----
        storage_1 = SQLiteStorage(db_path=test_db_path)
        runtime_1 = AgentRuntime(storage=storage_1)

        # 写入窗口 window_work 对话与待办
        sess_1 = runtime_1.sessions.get_or_create(session_id="window_work", user_id="user_test")
        sess_1.context.add_user_message("Persistent question from process 1")
        sess_1.context.add_assistant_message(content="Persistent answer from process 1")

        todo_tool_1 = runtime_1.tools.get_tool("todo")
        assert isinstance(todo_tool_1, TodoTool)
        todo_tool_1.run(action="add", title="Task that must survive restart", session_id="window_work")

        # 模拟进程完全退出: 显式关闭连接
        storage_1.close()
        del runtime_1
        del storage_1
        gc.collect()

        # ---- 进程 2 (全新冷启动) ----
        storage_2 = SQLiteStorage(db_path=test_db_path)
        runtime_2 = AgentRuntime(storage=storage_2)

        # 验证会话自动从磁盘预热
        assert "window_work" in runtime_2.sessions.list_sessions()

        # 验证消息历史无损还原
        sess_2 = runtime_2.sessions.get("window_work")
        assert sess_2 is not None
        messages = sess_2.context.get_messages()
        history_contents = [m.content for m in messages if m.role == "user"]
        assert "Persistent question from process 1" in history_contents

        # 验证待办事项从磁盘无损还原
        todo_tool_2 = runtime_2.tools.get_tool("todo")
        todos = todo_tool_2.run(action="list", session_id="window_work")["todos"]
        assert len(todos) == 1
        assert todos[0]["title"] == "Task that must survive restart"

        storage_2.close()

    finally:
        gc.collect()
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except Exception:
                pass


def test_cascade_delete_cleans_todos_and_context():
    """
    痛点 4 验证: “删掉 session 不会清掉该窗口的 Todo”
    
    测试逻辑:
      在 window_alpha 下添加待办并生成对话；
      执行 runtime.delete_session('window_alpha')；
      断言:
        1. 会话被删除；
        2. 上下文被清空；
        3. 底层存储中 window_alpha 的 Todo 被彻底级联物理删除，绝不残留孤儿数据！
    """
    test_db_path = "data/test_cascade.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    try:
        storage = SQLiteStorage(db_path=test_db_path)
        runtime = AgentRuntime(storage=storage)

        # 创建会话并添加待办
        runtime.sessions.get_or_create(session_id="window_alpha")
        todo_tool = runtime.tools.get_tool("todo")
        assert isinstance(todo_tool, TodoTool)
        todo_tool.run(action="add", title="Alpha Todo 1", session_id="window_alpha")
        todo_tool.run(action="add", title="Alpha Todo 2", session_id="window_alpha")

        # 确认添加成功
        assert len(todo_tool.run(action="list", session_id="window_alpha")["todos"]) == 2

        # 执行级联删除
        success = runtime.delete_session("window_alpha")
        assert success is True

        # 核心断言: Todo 必须被连带清空！
        todos_after_delete = todo_tool.run(action="list", session_id="window_alpha")["todos"]
        assert len(todos_after_delete) == 0

        # 会话本身亦不存在
        assert runtime.sessions.get("window_alpha") is None
        assert "window_alpha" not in runtime.sessions.list_sessions()

        storage.close()

    finally:
        gc.collect()
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except Exception:
                pass


def test_session_state_machine_transition():
    """
    痛点 5 验证: “SessionState 没有参与运行”
    
    测试逻辑:
      检验 SessionState 在初始化、运行时与完成时的状态机生命周期：
        初始态 -> running 态 (current_step 自增) -> 完成态 (turn_count 累加，恢复 idle)。
    """
    storage = SQLiteStorage(db_path=":memory:")
    session_mgr = SessionManager(storage=storage)
    session = session_mgr.get_or_create("state_test_window")

    # 1. 验证初始状态
    assert session.state.status == "idle"
    assert session.state.turn_count == 0
    assert session.state.current_step == 0

    # 2. 模拟进入运行态
    session.set_status("running")
    session.set_current_step(1)
    assert session.state.status == "running"
    assert session.state.current_step == 1

    # 3. 模拟完成一轮
    session.set_status("idle")
    session.increment_turn()
    assert session.state.status == "idle"
    assert session.state.turn_count == 1

    # 4. 验证持久化落盘状态一致
    persisted = storage.get_session_state("state_test_window")
    assert persisted is not None
    assert persisted.status == "idle"
    assert persisted.turn_count == 1

