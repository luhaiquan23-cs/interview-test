"""
Session 会话隔离与生命周期单元测试 (Unit Tests for Session)
----------------------------------------------------------------------
覆盖范围:
  1. 多窗口会话隔离验证: 模拟用户 A 窗口 1（查天气记待办）与窗口 2（写周报记待办）的独立性；
  2. 会话断点恢复 (Reconnect & Continue): 验证在不同窗口间自由切换时上下文不丢失、不交叉；
  3. 会话生命周期管理 (Lifecycle): 增删查、按用户聚合与清理。
"""

import pytest
from agent.core.session import SessionManager


def test_session_isolation():
    """
    笔试题重点场景测试:
      用户 A 开启窗口 1: 查天气记待办
      用户 A 开启窗口 2: 写周报记待办
      验证窗口 1 与窗口 2 的上下文历史绝对隔离，彼此无法看到对方的对话内容。
    """
    manager = SessionManager()

    # 1. 用户 A 在窗口 1 发送消息
    s1 = manager.get_or_create(session_id="window_1", user_id="user_A")
    s1.context.add_user_message("Check Guangzhou weather and add a todo: buy umbrella")
    s1.context.add_assistant_message("Guangzhou is sunny, but I added 'buy umbrella' to your todo.")

    # 2. 用户 A 在窗口 2 发送完全不同主题的消息
    s2 = manager.get_or_create(session_id="window_2", user_id="user_A")
    s2.context.add_user_message("Write weekly report and add a todo: submit report Friday")
    s2.context.add_assistant_message("Drafted your report and added 'submit report Friday' to todo.")

    # 3. 分别提取窗口 1 与窗口 2 的消息记录
    s1_history = [m.content for m in s1.context.get_messages() if m.role == "user"]
    s2_history = [m.content for m in s2.context.get_messages() if m.role == "user"]

    # 4. 断言隔离性
    assert len(s1_history) == 1
    assert "Guangzhou weather" in s1_history[0]
    assert "weekly report" not in s1_history[0]

    assert len(s2_history) == 1
    assert "weekly report" in s2_history[0]
    assert "Guangzhou weather" not in s2_history[0]


def test_session_reconnect_and_continue():
    """
    测试窗口会话恢复:
    用户在窗口 1 聊完后切到窗口 2，再切回窗口 1，依然能够获取先前的完整会话上下文。
    """
    manager = SessionManager()

    # 窗口 1 写入
    s1 = manager.get_or_create(session_id="window_1")
    s1.context.add_user_message("Hello from window 1")

    # 切换至窗口 2 写入
    s2 = manager.get_or_create(session_id="window_2")
    s2.context.add_user_message("Hello from window 2")

    # 切回窗口 1
    s1_reconnected = manager.get_or_create(session_id="window_1")
    assert s1_reconnected is s1
    assert len(s1_reconnected.context.get_messages()) == 2
    assert s1_reconnected.context.get_messages()[1].content == "Hello from window 1"


def test_session_lifecycle():
    """测试会话管理器基础生命周期（按用户检索、销毁删除）"""
    manager = SessionManager()
    s1 = manager.get_or_create("s1", user_id="user_1")
    s2 = manager.get_or_create("s2", user_id="user_1")
    s3 = manager.get_or_create("s3", user_id="user_2")

    # 按用户过滤会话列表
    assert set(manager.list_sessions(user_id="user_1")) == {"s1", "s2"}
    assert manager.list_sessions(user_id="user_2") == ["s3"]

    # 删除指定会话
    assert manager.delete("s1") is True
    assert manager.get("s1") is None
