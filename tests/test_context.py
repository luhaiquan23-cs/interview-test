"""
Context 上下文管理器单元测试 (Unit Tests for Context)
----------------------------------------------------------------------
覆盖范围:
  1. 基础上下文装配与轮次统计 (Turn Count);
  2. 长期记忆 (Memory) 动态注入与系统提示词 (System Prompt) 联动;
  3. 基础上下文压缩 (Compaction): 超过阈值时触发滑动窗口切分与结构化摘要提炼;
  4. 多轮追问支持 (Follow-up Continuity): 验证上下文对追问语境的保真度。
"""

import pytest
from agent.core.context import ContextManager
from agent.core.schema import Message


def test_context_basic():
    """测试基础消息添加、轮次统计与初始 System Prompt 校验"""
    ctx = ContextManager(max_turns=5, compression_threshold=10)
    assert len(ctx.get_messages()) == 1
    assert ctx.get_messages()[0].role == "system"

    ctx.add_user_message("Hello Agent")
    ctx.add_assistant_message(content="Hello! How can I help you today?")
    assert ctx.count_turns() == 1
    assert len(ctx.get_messages()) == 3


def test_context_memory_injection():
    """测试长期记忆 (Long-term Memory) 注入至 System Prompt 的机制"""
    ctx = ContextManager()
    ctx.add_memory("User lives in Guangzhou.")
    ctx.add_memory("User prefers responses in Chinese.")

    sys_content = ctx.get_messages()[0].content
    # 验证系统提示词中包含长效记忆标签与具体偏好内容
    assert "Guangzhou" in sys_content
    assert "Chinese" in sys_content
    assert "[Long-term Memory / Known User Context]" in sys_content


def test_context_compaction():
    """
    测试滑动窗口基础压缩 (Basic Context Compaction):
    设置 threshold=4, keep_recent=2。当消息数超标时，远期消息压缩为摘要，近期消息原样保留。
    """
    ctx = ContextManager(compression_threshold=4, keep_recent_messages=2)

    # 连续追加 5 条对话消息，触发压缩机制
    ctx.add_user_message("Message 1")
    ctx.add_assistant_message("Reply 1")
    ctx.add_user_message("Message 2")
    ctx.add_assistant_message("Reply 2")
    ctx.add_user_message("Message 3")  # 超过阈值 4 -> 自动触发压缩

    messages = ctx.get_messages()
    # 验证第 0 条始终是包含压缩摘要的系统消息
    assert messages[0].role == "system"
    assert "Summary of Prior Conversation" in messages[0].content
    assert "User asked: Message 1" in messages[0].content

    # 验证最近的 Message 3 被完整保留在近期上下文中
    recent_contents = [m.content for m in messages[1:]]
    assert "Message 3" in recent_contents


def test_context_followup_preservation():
    """
    测试多轮追问的上下文完整性:
    验证前序问答（如计算 20+30=50）与后序追问（乘 2）的上下文连续性。
    """
    ctx = ContextManager()
    ctx.add_user_message("What is 20 + 30?")
    ctx.add_assistant_message(content="It is 50.")
    ctx.add_user_message("Now multiply that by 2")

    msgs = ctx.get_messages()
    assert len(msgs) == 4
    assert msgs[1].content == "What is 20 + 30?"
    assert msgs[2].content == "It is 50."
    assert msgs[3].content == "Now multiply that by 2"
