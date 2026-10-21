"""
Context 上下文管理器
----------------------------------------------------------------------
负责管理单个 Session 会话的消息历史、最大轮次限制、多轮追问保真、长期记忆注入以及滑动窗口基础压缩。

设计核心原则:
  1. 系统提示词恒定保真 (System Invariant): 核心角色设定与全局长效记忆不受上下文截断影响；
  2. 追问保真支持 (Follow-up Continuity): 保留最近 $k$ 条对话及工具返回值，支持纯对话追问与带工具结果追问；
  3. 基础压缩机制 (Compaction): 超过压缩阈值时，自动将远期对话提炼为结构化背景摘要注入系统提示词中，避免 Context 爆炸。
"""

from typing import List, Dict, Any, Optional
from .schema import Message


class ContextManager:
    """
    会话上下文管理器。
    
    参数:
      - system_prompt: 系统提示词（人设与全局规范）
      - max_turns: 建议的最大对话轮次
      - compression_threshold: 触发基础上下文压缩的消息数阈值（默认 12 条）
      - keep_recent_messages: 压缩时完整保留的高保真近期消息条数（默认 6 条）
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an intelligent, capable, and helpful AI Agent.\n"
        "You have access to a suite of tools (calculator, search, todo, weather).\n"
        "Guidelines:\n"
        "1. When a task requires external data, math calculation, or updating personal todos, proactively call tools.\n"
        "2. Think step-by-step before deciding to call tools or providing a final answer.\n"
        "3. Support multi-turn follow-up queries naturally by referencing past dialogue and tool results.\n"
        "4. Always be concise, helpful, and polite."
    )

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
        compression_threshold: int = 12,
        keep_recent_messages: int = 6,
    ):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_turns = max_turns
        self.compression_threshold = compression_threshold
        self.keep_recent_messages = keep_recent_messages

        # 内部维护的序列化消息列表
        self.messages: List[Message] = []
        # 远期对话提取的背景摘要
        self._summary: Optional[str] = None
        # 用户画像或长期记忆列表
        self._memories: List[str] = []

        # 初始化带有系统提示词的消息列表
        self.reset()

    def reset(self):
        """重置上下文至初始状态（保留系统提示词）"""
        self.messages = [Message(role="system", content=self._build_system_content())]
        self._summary = None

    def add_memory(self, memory_text: str):
        """
        注入一条用户长期记忆（如城市偏好、身份角色或习惯）。
        此记忆将动态更新至系统提示词中，确保全场对话均能召回该背景。
        """
        if memory_text not in self._memories:
            self._memories.append(memory_text)
            self._refresh_system_message()

    def _build_system_content(self) -> str:
        """
        组合生成完整的 System Prompt 内容：
          [核心系统指令]
          + [长效记忆/用户画像] (如果有)
          + [远期历史对话摘要] (如果已触发压缩)
        """
        parts = [self.system_prompt]
        if self._memories:
            memory_block = "\n".join(f"- {m}" for m in self._memories)
            parts.append(f"\n[Long-term Memory / Known User Context]:\n{memory_block}")
        if self._summary:
            parts.append(f"\n[Summary of Prior Conversation]:\n{self._summary}")
        return "\n".join(parts)

    def _refresh_system_message(self):
        """刷新索引为 0 的 system 消息内容"""
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = self._build_system_content()
        else:
            self.messages.insert(0, Message(role="system", content=self._build_system_content()))

    def add_user_message(self, content: str):
        """
        写入用户新指令，并在满足条件时自动触发上下文压缩检查。
        """
        self.messages.append(Message(role="user", content=content))
        self._check_and_compact()

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        thought: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        写入模型回复（包含思考过程 thought 或工具调用请求 tool_calls）。
        """
        self.messages.append(
            Message(
                role="assistant",
                content=content,
                thought=thought,
                tool_calls=tool_calls,
            )
        )

    def add_tool_message(self, tool_call_id: str, name: str, content: str):
        """
        写入工具执行结果（role="tool"），供模型在下一轮决策中观察。
        """
        self.messages.append(
            Message(
                role="tool",
                name=name,
                tool_call_id=tool_call_id,
                content=content,
            )
        )

    def get_messages(self) -> List[Message]:
        """返回当前准备提交给 LLM 的全量消息列表快照"""
        return list(self.messages)

    def count_turns(self) -> int:
        """计算当前对话中用户交互的实际轮次数"""
        return sum(1 for m in self.messages if m.role == "user")

    def _check_and_compact(self):
        """
        基础上下文压缩机制 (Basic Context Compaction):
        ------------------------------------------------------------------
        当非系统消息总数超过 compression_threshold 时触发压缩：
          1. 切分：分为需压缩的“远期消息”和需原样保留的“近期消息”；
          2. 提取：将远期消息中关键的用户意图、工具调用名与结果概况提取为结构化摘要；
          3. 合并：将结构化摘要追加至 System Prompt 的 [Summary of Prior Conversation]；
          4. 保留：完整保留最近的 keep_recent_messages 条消息（确保当前追问的语境连续与高保真）。
        """
        # 排除首部的 system 消息
        non_system_messages = self.messages[1:] if self.messages and self.messages[0].role == "system" else self.messages

        if len(non_system_messages) <= self.compression_threshold:
            return

        # 计算切分点
        split_idx = max(0, len(non_system_messages) - self.keep_recent_messages)
        to_compress = non_system_messages[:split_idx]
        to_keep = non_system_messages[split_idx:]

        # 结构化提取远期消息要点（避免发起额外的异步 LLM 摘要调用以节省开销与时延）
        summary_lines = []
        for msg in to_compress:
            if msg.role == "user":
                summary_lines.append(f"User asked: {msg.content}")
            elif msg.role == "assistant":
                if msg.tool_calls:
                    calls_desc = ", ".join(tc.get("function", {}).get("name", "tool") for tc in msg.tool_calls)
                    summary_lines.append(f"Agent called tools: [{calls_desc}]")
                if msg.content:
                    summary_lines.append(f"Agent responded: {msg.content[:80]}...")
            elif msg.role == "tool":
                summary_lines.append(f"Tool [{msg.name}] returned: {msg.content[:60]}...")

        new_summary_text = "\n".join(summary_lines)
        if self._summary:
            self._summary = f"{self._summary}\n{new_summary_text}"
        else:
            self._summary = new_summary_text

        # 重新装配消息列表：更新后的系统消息（含摘要） + 最近的高保真上下文
        self.messages = [Message(role="system", content=self._build_system_content())] + to_keep
