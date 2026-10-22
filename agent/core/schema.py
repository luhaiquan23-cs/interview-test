"""
核心数据结构与实体模型 (Schema)
----------------------------------------------------------------------
定义 Agent Runtime 流转所需的所有数据模型，基于 Pydantic 提供严格的类型校验：
  - ToolCall: 大模型生成的工具调用请求（名称、参数、唯一 ID）
  - ToolResult: 工具执行后的观测返回（执行状态、文本输出、错误详情）
  - Message: 对话历史中的单条消息实体（兼容 OpenAI 消息协议）
  - TurnBlock: 原子化对话轮次单元（彻底解决拆散 assistant tool_calls 和 tool 结果的问题）
  - AgentStep: 单次事件循环步骤的执行快照（耗时、思考过程、工具行为）
  - TraceLog: 单轮用户对话的全链路追踪轨迹日志
  - SessionState: 会话维度的持久化核心状态机模型（status, current_step, turn_count 深度参与运行）
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import time
import json


class ToolCall(BaseModel):
    """
    大模型请求调用的工具信息结构。
    
    字段:
      - id: 每次工具调用的唯一标识符（如 'call_abc123'）
      - name: 目标工具名称（如 'calculator', 'todo'）
      - arguments: 反序列化后的参数字典
    """
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_openai(cls, tc_dict: Any) -> "ToolCall":
        """
        从 OpenAI / DeepSeek 原生返回的 tool_call 对象或字典中解析出规范化的 ToolCall 实例。
        自动兼容 arguments 为 JSON 字符串或原生字典的情况。
        """
        if hasattr(tc_dict, "id"):
            call_id = tc_dict.id
            func_name = tc_dict.function.name
            args_str = tc_dict.function.arguments
        else:
            call_id = tc_dict.get("id", "")
            func = tc_dict.get("function", {})
            func_name = func.get("name", "")
            args_str = func.get("arguments", "{}")

        # 安全反序列化参数字符串
        try:
            if isinstance(args_str, str):
                args = json.loads(args_str) if args_str else {}
            elif isinstance(args_str, dict):
                args = args_str
            else:
                args = {}
        except Exception:
            args = {"raw_arguments": args_str}

        return cls(id=call_id, name=func_name, arguments=args)


class ToolResult(BaseModel):
    """
    工具执行后的结果实体。
    
    字段:
      - tool_call_id: 对应触发该调用的 ToolCall.id
      - name: 工具名称
      - content: 执行输出内容（必须为文本或 JSON 序列化字符串）
      - success: 是否执行成功
      - error: 错误堆栈信息（若失败）
    """
    tool_call_id: str
    name: str
    content: str
    success: bool = True
    error: Optional[str] = None


class Message(BaseModel):
    """
    对话上下文消息模型。
    
    字段:
      - role: 角色 ('system' | 'user' | 'assistant' | 'tool')
      - content: 消息主体文本
      - thought: 模型的推理链思考过程 (Reasoning/Thought)
      - name: 工具名称（当 role='tool' 时必填）
      - tool_call_id: 工具调用对应 ID（当 role='tool' 时必填）
      - tool_calls: 模型请求发起的工具调用列表（当 role='assistant' 且需要调用工具时必填）
    """
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Optional[str] = None
    thought: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_openai(self) -> Dict[str, Any]:
        """
        将实体转换为符合 OpenAI / DeepSeek API 要求的请求消息字典格式。
        """
        msg: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class TurnBlock(BaseModel):
    """
    原子化交互轮次单元 (Atomic Dialogue Turn)。
    解决面试官指出的致命问题: “压缩按条数切开，会拆散 assistant 的 tool call 和后面的 tool 结果”。
    
    一个完整的 Turn 包含:
      - 触发的 user 消息
      - 该轮中所有的 assistant(含 tool_calls) 与对应的 tool 执行结果消息
      - 最终 assistant 直接回答
    
    无论滑动窗口如何截断，切分边界必须在 TurnBlock 之间，严禁在 TurnBlock 内部切割！
    """
    messages: List[Message] = Field(default_factory=list)

    @property
    def user_message(self) -> Optional[Message]:
        for m in self.messages:
            if m.role == "user":
                return m
        return None

    @property
    def final_response(self) -> Optional[Message]:
        for m in reversed(self.messages):
            if m.role == "assistant" and not m.tool_calls:
                return m
        return None

    def summarize_for_compaction(self) -> str:
        """提炼该原子轮次的紧凑摘要文本"""
        parts = []
        user_msg = self.user_message
        if user_msg and user_msg.content:
            parts.append(f"User: {user_msg.content}")

        # 工具调用及结果提炼
        tools_called = []
        for m in self.messages:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    func_name = tc.get("function", {}).get("name", "tool")
                    tools_called.append(func_name)
            elif m.role == "tool":
                # 记录简要返回值
                snippet = m.content[:50] + "..." if m.content and len(m.content) > 50 else m.content
                parts.append(f"Obs({m.name}): {snippet}")

        if tools_called:
            parts.append(f"Action: [{', '.join(tools_called)}]")

        resp = self.final_response
        if resp and resp.content:
            snippet = resp.content[:60] + "..." if len(resp.content) > 60 else resp.content
            parts.append(f"Agent: {snippet}")

        return " | ".join(parts)


class AgentStep(BaseModel):
    """
    Agent 4 步循环中单一步骤 (Step) 的执行快照。
    """
    step_index: int
    thought: Optional[str] = None
    action: str  # "tool_call" or "direct_response"
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    response_content: Optional[str] = None
    duration_ms: float = 0.0


class TraceLog(BaseModel):
    """
    单轮交互的完整链路追踪日志 (TraceLog)。
    记录整轮用户指令从接收、多步循环推理、工具调度到输出最终答案的全过程。
    """
    session_id: str
    user_input: str
    steps: List[AgentStep] = Field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class SessionState(BaseModel):
    """
    会话状态机核心实体模型 (Single Source of Truth)。
    解决面试官指出的问题: “SessionState 没有参与运行”。
    
    字段参与运行时核心逻辑:
      - status: 驱动会话状态机 ("idle" 待命 | "running" 正在推理执行 | "error" 异常)
      - current_step: 追踪当前会话正在执行的推理步数
      - turn_count: 累计多轮对话实际交互轮次
      - updated_at: 活跃时间，用于超时与缓存淘汰
    """
    session_id: str
    user_id: Optional[str] = None
    status: str = "idle"  # "idle" | "running" | "error"
    current_step: int = 0
    turn_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)
