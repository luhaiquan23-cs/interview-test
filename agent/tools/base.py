"""
工具系统基础规范与注册中心
----------------------------------------------------------------------
负责定义工具基类 (BaseTool)、函数自动转换装饰器 (@tool) 以及工具注册中心 (ToolRegistry)。
满足题目要求：
  - 需实现工具注册机制（每个工具包含名称、描述、参数 Schema）；
  - LLM 基于 Schema 自主决策调用；
  - 具备执行异常捕获与容错包装，保证 Agent 主循环不崩溃。
"""

import inspect
import json
import traceback
from typing import Dict, Any, Callable, Optional, List, Type
from pydantic import BaseModel
from agent.core.schema import ToolCall, ToolResult


class BaseTool:
    """
    所有 Agent 工具的基类。
    
    子类必须声明:
      - name: 工具英文唯一标识符（如 'calculator', 'todo'）
      - description: 工具功能详述（供大模型理解其使用场景）
      - parameters: 符合 JSON Schema 规范的参数定义字典
    """
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    def run(self, **kwargs) -> Any:
        """执行具体工具逻辑，由子类实现"""
        raise NotImplementedError("Subclasses must implement run()")

    def to_schema(self) -> Dict[str, Any]:
        """
        转换为 OpenAI / DeepSeek 原生兼容的 Function Calling 格式。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionTool(BaseTool):
    """
    从普通 Python 函数动态推导参数并包装而成的 Tool 实例。
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        self.parameters = parameters or self._infer_parameters(func)

    def run(self, **kwargs) -> Any:
        return self.func(**kwargs)

    def _infer_parameters(self, func: Callable) -> Dict[str, Any]:
        """通过函数类型注解动态反射生成 JSON Schema 参数规范"""
        sig = inspect.signature(func)
        properties = {}
        required = []

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                param_type = type_map.get(param.annotation, "string")

            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}",
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


def tool(name: Optional[str] = None, description: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
    """
    装饰器: 将任意 Python 函数快速声明注册为 Agent 可调用的工具。
    """
    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(func, name=name, description=description, parameters=parameters)
    return decorator


class ToolRegistry:
    """
    工具注册与调度中心。
    负责聚合可用工具、输出给大模型的 Schemas 列表、以及安全分发执行具体工具。
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool_or_func: Any) -> None:
        """注册一个 BaseTool 实例或带有类型注解的普通函数"""
        if isinstance(tool_or_func, BaseTool):
            self._tools[tool_or_func.name] = tool_or_func
        elif callable(tool_or_func):
            ft = FunctionTool(tool_or_func)
            self._tools[ft.name] = ft
        else:
            raise TypeError(f"Cannot register object of type {type(tool_or_func)}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具实例"""
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        获取当前已注册全部工具的 JSON Schema 定义列表。
        直接传入 LLM 请求中的 `tools` 参数。
        """
        return [tool.to_schema() for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        安全分发并执行工具调用。
        
        容错设计:
          1. 未找到工具时，不抛异常崩溃，而是返回包含所有可用工具列表的错误提示；
          2. 工具执行抛出任意异常时，捕获异常堆栈包装为 ToolResult(success=False)；
          3. 这样可作为 Observation 返回给 LLM，允许大模型感知错误并自主重试或纠正参数。
        """
        tool_obj = self._tools.get(tool_call.name)
        if not tool_obj:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Tool '{tool_call.name}' is not registered. Available tools: {list(self._tools.keys())}",
                success=False,
                error="ToolNotFound",
            )

        try:
            # 传入参数执行具体工具业务
            raw_result = tool_obj.run(**tool_call.arguments)
            # 序列化为规范字符串
            if isinstance(raw_result, (dict, list)):
                content = json.dumps(raw_result, ensure_ascii=False)
            else:
                content = str(raw_result)

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=content,
                success=True,
            )
        except Exception as e:
            # 捕获异常，自愈式反馈
            err_msg = f"Tool '{tool_call.name}' execution failed with error: {str(e)}"
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=err_msg,
                success=False,
                error=traceback.format_exc(),
            )
