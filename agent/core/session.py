"""
Session 多会话管理
----------------------------------------------------------------------
负责管理多窗口（Multi-Window）并发独立的对话会话。
对应笔试题核心场景：
  - 用户 A 开了窗口 1：让 Agent 查天气记待办
  - 用户 A 开了窗口 2：让 Agent 写周报记待办
  - 窗口 1 与窗口 2 必须是相互独立的 Session，用户随时切回任一窗口继续聊，彼此上下文和待办互不影响。
"""

import time
from typing import Dict, Any, Optional, List
from .schema import SessionState
from .context import ContextManager


class Session:
    """
    单个独立会话实例。
    每个 Session 拥有独立的 ContextManager 实例（独享历史消息、长期记忆与压缩状态）。
    """

    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        context_manager: Optional[ContextManager] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        # 独享独立的上下文管理器，彻底隔离上下文污染
        self.context = context_manager or ContextManager()
        self.created_at = time.time()
        self.updated_at = time.time()
        self.metadata = metadata or {}

    def touch(self):
        """更新最后一次活跃时间戳"""
        self.updated_at = time.time()

    def to_state(self) -> SessionState:
        """导出当前会话的元数据状态"""
        return SessionState(
            session_id=self.session_id,
            user_id=self.user_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            metadata=self.metadata,
        )


class SessionManager:
    """
    多会话隔离管理器。
    负责维护全生命周期的 Session 实例池，提供安全的会话获取、隔离与切换机制。
    """

    def __init__(self):
        # 内存中维护的会话映射字典：session_id -> Session
        self._sessions: Dict[str, Session] = {}

    def get_or_create(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Session:
        """
        获取已有会话；若不存在则自动初始化一个全新的隔离会话。

        参数:
          - session_id: 会话标识符（如 'window_1', 'window_2'）
          - user_id: 用户标识符
          - system_prompt: 自定义系统提示词（可选）
        """
        if session_id not in self._sessions:
            ctx = ContextManager(system_prompt=system_prompt) if system_prompt else ContextManager()
            self._sessions[session_id] = Session(
                session_id=session_id,
                user_id=user_id,
                context_manager=ctx,
            )
        else:
            self._sessions[session_id].touch()
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        """查询指定 session_id 的会话实例"""
        return self._sessions.get(session_id)

    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """列出当前存在的所有会话 ID（支持按 user_id 过滤）"""
        if user_id:
            return [
                sid for sid, s in self._sessions.items()
                if s.user_id == user_id
            ]
        return list(self._sessions.keys())

    def delete(self, session_id: str) -> bool:
        """销毁并清理指定会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def clear(self):
        """清空所有会话"""
        self._sessions.clear()
