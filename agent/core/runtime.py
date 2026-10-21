"""
Agent Runtime 核心实现
----------------------------------------------------------------------
该模块负责驱动整个 Agent 的事件驱动循环（4-Step Loop），是 Agent 的决策与执行中枢。
严格遵循题目要求：
  - Step 1: 接收用户输入（并关联独立会话 Session 上下文）
  - Step 2: 调用 LLM 判断是直接回复，还是调用工具（解析思考过程与函数调用）
  - Step 3: 安全分发并执行工具调用（异常捕获与自纠错容错）
  - Step 4: 根据工具执行结果判断是继续循环还是输出最终答案给用户（最大步数防死循环熔断）
"""

import time
import json
from typing import Optional, List, Dict, Any, Callable
from .schema import (
    Message,
    ToolCall,
    ToolResult,
    AgentStep,
    TraceLog,
)
from .llm import DeepSeekClient
from .context import ContextManager
from .session import SessionManager, Session
from agent.tools.base import ToolRegistry
from agent.tools.builtin import (
    CalculatorTool,
    SearchTool,
    TodoTool,
    WeatherTool,
    TodoStore,
)


class AgentRuntime:
    """
    零框架 Minimal Agent Runtime。
    
    属性说明:
      - llm: DeepSeek 大模型客户端，负责推理与 Function Calling 决策
      - tools: 工具注册中心，维护所有可用工具与其 JSON Schema 定义
      - sessions: 多会话隔离管理器，支持多窗口独立对话与状态隔离
      - max_steps_per_turn: 单轮对话内允许的最大工具迭代步数（防止模型陷入死循环）
      - trace_callback: 步进式执行日志回调函数（用于实时控制台渲染或链路追踪）
    """

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        session_manager: Optional[SessionManager] = None,
        max_steps_per_turn: int = 6,
        trace_callback: Optional[Callable[[AgentStep], None]] = None,
    ):
        self.llm = llm_client or DeepSeekClient()
        self.tools = tool_registry or self._create_default_registry()
        self.sessions = session_manager or SessionManager()
        self.max_steps_per_turn = max_steps_per_turn
        self.trace_callback = trace_callback

    def _create_default_registry(self) -> ToolRegistry:
        """
        初始化默认工具注册中心。
        内置实现的 4 大工具：
          1. calculator (安全数学运算)
          2. search (知识/全网检索)
          3. todo (待办事项管理，题目必须实现)
          4. weather (天气与气温查询)
        """
        registry = ToolRegistry()
        todo_store = TodoStore()

        registry.register(CalculatorTool())
        registry.register(SearchTool())
        registry.register(TodoTool(store=todo_store))
        registry.register(WeatherTool())
        return registry

    def run(
        self,
        user_input: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
    ) -> (str, TraceLog):
        """
        执行一轮完整的用户交互循环（4-Step Loop）。

        参数:
          - user_input: 用户输入的自然语言指令
          - session_id: 当前会话窗口 ID（如 'window_1', 'window_2'，确保多窗口互不干扰）
          - user_id: 用户唯一标识符（可选）

        返回:
          - final_answer: Agent 返回给用户的最终文本答复
          - trace: 本轮交互的全量调用链路跟踪日志（TraceLog）
        """
        start_time = time.time()
        trace = TraceLog(session_id=session_id, user_input=user_input)

        # 1. 获取或创建独立会话（确保 Window 1 与 Window 2 状态严格隔离）
        session = self.sessions.get_or_create(session_id=session_id, user_id=user_id)

        # 2. 将当前活跃会话 ID 注入支持会话隔离的工具（如 TodoTool）
        todo_tool = self.tools.get_tool("todo")
        if isinstance(todo_tool, TodoTool):
            todo_tool.set_session_id(session_id)

        # ----------------------------------------------------
        # Step 1: 接收用户输入并写入当前 Session 上下文
        # ----------------------------------------------------
        session.context.add_user_message(user_input)

        final_answer = ""
        step_index = 0

        # 进入自主决策与工具调度循环
        while step_index < self.max_steps_per_turn:
            step_index += 1
            step_start = time.time()

            # ----------------------------------------------------
            # Step 2: 经由 LLM 判断是直接回复，还是调用工具
            # ----------------------------------------------------
            messages = session.context.get_messages()
            tool_schemas = self.tools.get_schemas()

            try:
                # 调用 DeepSeek API，返回: (文本回复, 思考过程, 工具调用列表, 结束原因)
                content, thought, tool_calls, finish_reason = self.llm.generate(
                    messages=messages,
                    tools=tool_schemas,
                )
            except Exception as e:
                # 针对网络中断或 API 异常进行优雅降级
                err_msg = f"LLM generation failed: {str(e)}"
                trace.success = False
                trace.error = err_msg
                final_answer = f"Sorry, I encountered an internal error: {err_msg}"
                break

            # 分支 A: 模型决定直接回复（无工具调用）
            if not tool_calls:
                # 将 Assistant 回复追加至上下文，结束当前交互循环
                session.context.add_assistant_message(content=content, thought=thought)
                final_answer = content or ""
                step = AgentStep(
                    step_index=step_index,
                    thought=thought,
                    action="direct_response",
                    response_content=final_answer,
                    duration_ms=(time.time() - step_start) * 1000,
                )
                trace.steps.append(step)
                if self.trace_callback:
                    self.trace_callback(step)
                # 跳出循环，返回最终答案给用户
                break

            # 分支 B: 模型决定调用工具
            # ----------------------------------------------------
            # Step 3: 安全分发并执行工具调用
            # ----------------------------------------------------
            # 注意: DeepSeek / OpenAI API 规范要求 assistant 历史消息中 tool_calls[i].function.arguments
            # 必须为序列化后的 JSON 字符串，否则会触发 422 校验失败异常。
            openai_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        if isinstance(tc.arguments, dict)
                        else str(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]
            # 记录 Assistant 的工具调用意图
            session.context.add_assistant_message(
                content=content,
                thought=thought,
                tool_calls=openai_tool_calls,
            )

            # 遍历并安全执行所有请求的工具
            tool_results: List[ToolResult] = []
            for tc in tool_calls:
                res = self.tools.execute(tc)
                tool_results.append(res)
                # 将每个工具的执行结果以 role="tool" 写入上下文供后续轮次观察
                session.context.add_tool_message(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=res.content,
                )

            # 记录当前 Step 的 Trace 信息
            step = AgentStep(
                step_index=step_index,
                thought=thought,
                action="tool_call",
                tool_calls=tool_calls,
                tool_results=tool_results,
                response_content=content,
                duration_ms=(time.time() - step_start) * 1000,
            )
            trace.steps.append(step)
            if self.trace_callback:
                self.trace_callback(step)

            # ----------------------------------------------------
            # Step 4: 根据工具结果判断是继续 loop，还是返回结果给用户
            # ----------------------------------------------------
            # 此时工具结果已全部追加至上下文，while 循环自动进入下一轮迭代，
            # 再次触发 Step 2，由 LLM 根据工具执行结果评估是否继续调用其它工具或组织最终回复。

        # 熔断兜底：如果达到最大步数限制（例如 6 步）仍未生成直接回复，触发安全保护
        if not final_answer:
            final_answer = "I completed the tool executions, but reached the maximum reasoning steps limit."
            session.context.add_assistant_message(content=final_answer)

        # 汇总完整耗时并返回
        trace.final_answer = final_answer
        trace.total_duration_ms = (time.time() - start_time) * 1000
        return final_answer, trace

