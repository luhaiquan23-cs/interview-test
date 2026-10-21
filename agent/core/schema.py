"""
核心数据结构与实体模型 (Schema)
----------------------------------------------------------------------
定义 Agent Runtime 流转所需的所有数据模型，基于 Pydantic 提供严格的类型校验：
  - ToolCall: 大模型生成的工具调用请求（名称、参数、唯一 ID）
  - ToolResult: 工具执行后的观测返回（执行状态、文本输出、错误详情）
  - Message: 对话历史中的单条消息实体（兼容 OpenAI 消息协议）
  - AgentStep: 单次事件循环步骤的执行快照（耗时、思考过程、工具行为）
  - TraceLog: 单轮用户对话的全链路追踪轨迹日志
  - SessionState: 会话维度的持久化元数据
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


class AgentStep(BaseModel):
    """
    Agent 4 步循环中单一步骤 (Step) 的执行快照。
    
    字段:
      - step_index: 当前步骤序号（第 1 步、第 2 步等）
      - thought: 步骤开始前模型输出的思考内容
      - action: 动作类型 ('tool_call' 工具调用 或 'direct_response' 直接回复)
      - tool_calls: 本步骤调用的工具列表
      - tool_results: 本步骤所有工具执行后的结果
      - response_content: 模型生成的文本内容
      - duration_ms: 本步骤耗时（毫秒）
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
    会话状态与生命周期元数据。
    """
    session_id: str
    user_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)
