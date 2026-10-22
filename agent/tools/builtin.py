"""
内置工具集实现
----------------------------------------------------------------------
涵盖题目要求的必备与拓展工具：
  1. SafeCalculator / CalculatorTool: 安全数学计算器（基于 Python AST 语法树，严禁 eval()）
  2. SearchTool: 模拟知识与互联网搜索工具
  3. TodoStore / TodoTool: 待办事项 CRUD（必须实现，严格按 session_id 物理隔离）
  4. WeatherTool: 城市天气与气温查询工具
"""

import ast
import operator
import math
from typing import Dict, Any, List, Optional
from .base import BaseTool


class SafeCalculator:
    """
    基于 Python AST 抽象语法树的安全数学表达式计算器。
    
    安全设计:
      - 绝不使用 Python 原生 `eval()`，彻底杜绝任意代码执行（RCE）与系统提权漏洞；
      - 严格采用白名单机制：只允许指定的数学二元运算符、一元运算符以及安全的内置数学函数；
      - 任何未知节点、变量访问或系统调用（如 __import__, open, os）均会在 AST 解析期直接拦截。
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }

    @classmethod
    def evaluate(cls, expression: str) -> float:
        """解析并安全求值数学表达式字符串"""
        expr_clean = expression.strip()
        try:
            node = ast.parse(expr_clean, mode="eval").body
            return cls._eval_node(node)
        except Exception as e:
            raise ValueError(f"Invalid math expression '{expression}': {str(e)}")

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Any:
        """递归遍历 AST 语法节点求值"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            op_func = cls.OPERATORS.get(type(node.op))
            if not op_func:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            op_func = cls.OPERATORS.get(type(node.op))
            if not op_func:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op_func(operand)
        elif isinstance(node, ast.Call):
            # 仅允许调用预置在 FUNCTIONS 白名单中的数学函数
            if isinstance(node.func, ast.Name) and node.func.id in cls.FUNCTIONS:
                func = cls.FUNCTIONS[node.func.id]
                args = [cls._eval_node(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Unsupported function call in math expression: {ast.dump(node.func)}")
        elif isinstance(node, ast.Name):
            if node.id in cls.FUNCTIONS:
                return cls.FUNCTIONS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")


class CalculatorTool(BaseTool):
    """
    数学计算工具。
    供 Agent 在面对数值运算、复合公式计算时主动调用。
    """
    name = "calculator"
    description = "Compute mathematical expressions safely (e.g., '128 * 45', 'sqrt(144) + 10', '2 ** 8')."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical formula to compute, e.g. '15 * (24 + 18)' or 'sqrt(256)'.",
            }
        },
        "required": ["expression"],
    }

    def run(self, expression: str, **kwargs) -> Dict[str, Any]:
        result = SafeCalculator.evaluate(expression)
        return {"expression": expression, "result": result}


class SearchTool(BaseTool):
    """
    模拟搜索工具（可根据题目要求 mock 或接入真实网络）。
    内置真实场景相关的领域知识库，以便快速进行 Agent 行为验证。
    """
    name = "search"
    description = "Search the web or internal knowledge base for information, news, docs, or facts."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            }
        },
        "required": ["query"],
    }

    # 预置高质量检索语料
    MOCK_KNOWLEDGE = {
        "agent": "An AI Agent is an autonomous entity that perceives its environment through sensors/inputs, reasons about actions, and interacts using tools/actuators to achieve goals.",
        "deepseek": "DeepSeek is an advanced AI research organization known for DeepSeek-V3 and DeepSeek-R1 reasoning models.",
        "weather": "Weather forecasting analyzes meteorological data such as atmospheric pressure, temperature, humidity, and wind velocity.",
        "weekly report": "Weekly report guidelines: 1. Completed tasks; 2. Ongoing initiatives; 3. Blockers/risks; 4. Next week's key deliverables.",
        "python": "Python 3.13 was released in October 2024, introducing experimental free-threaded mode (no-GIL) and a JIT compiler.",
    }

    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        query_lower = query.lower()
        matched_results = []

        # 关键词与倒排匹配
        for key, snippet in self.MOCK_KNOWLEDGE.items():
            if key in query_lower or any(word in snippet.lower() for word in query_lower.split()):
                matched_results.append({
                    "title": f"Result for '{key}'",
                    "snippet": snippet,
                    "url": f"https://mocksearch.local/articles/{key}",
                })

        # 泛化兜底命中
        if not matched_results:
            matched_results.append({
                "title": f"Search summary for '{query}'",
                "snippet": f"Found relevant information regarding '{query}'. Overall industry consensus confirms standard operational best practices.",
                "url": f"https://mocksearch.local/search?q={query}",
            })

        return {"query": query, "results": matched_results}


class TodoStore:
    """
    待办事项持久化存储引擎 (v2 持久化增强版)。
    核心特性:
      1. 默认对接 SQLiteStorage 持久化落盘，进程重启后数据无损恢复；
      2. 强制按 session_id 进行物理分桶隔离，避免不同窗口之间的待办交叉污染；
      3. 支持 Session 销毁时的级联物理删除，彻底避免孤儿数据。
    """

    def __init__(self, storage: Optional[Any] = None):
        self.storage = storage
        # 内存回退字典（若未注入 SQLiteStorage 引擎时使用）
        self._todos: Dict[str, List[Dict[str, Any]]] = {}
        self._counter: int = 1

    def add(self, session_id: str, title: str, priority: str = "normal") -> Dict[str, Any]:
        """为指定会话添加一条待办"""
        if self.storage:
            return self.storage.add_todo(session_id, title=title, priority=priority)

        item = {
            "id": self._counter,
            "title": title,
            "priority": priority,
            "completed": False,
        }
        self._counter += 1
        if session_id not in self._todos:
            self._todos[session_id] = []
        self._todos[session_id].append(item)
        return item

    def list(self, session_id: str) -> List[Dict[str, Any]]:
        """获取指定会话的所有待办清单"""
        if self.storage:
            return self.storage.list_todos(session_id)
        return self._todos.get(session_id, [])

    def complete(self, session_id: str, todo_id: int) -> Optional[Dict[str, Any]]:
        """将指定会话中的某项待办标记为已完成"""
        if self.storage:
            return self.storage.complete_todo(session_id, todo_id)

        for item in self._todos.get(session_id, []):
            if item["id"] == todo_id:
                item["completed"] = True
                return item
        return None

    def delete(self, session_id: str, todo_id: int) -> bool:
        """删除指定会话中的某项待办"""
        if self.storage:
            return self.storage.delete_todo(session_id, todo_id)

        items = self._todos.get(session_id, [])
        for i, item in enumerate(items):
            if item["id"] == todo_id:
                items.pop(i)
                return True
        return False

    def clear(self, session_id: str) -> None:
        """清空指定会话的待办（级联清理）"""
        if self.storage:
            self.storage.clear_todos(session_id)
        else:
            self._todos[session_id] = []


class TodoTool(BaseTool):
    """
    待办事项管理工具（笔试题指定必须实现的工具）。
    支持完整 CRUD:
      - add: 新增待办
      - list: 查询当前窗口待办清单
      - complete: 标记待办已完成
      - delete: 删除待办
    """
    name = "todo"
    description = "Manage personal todo items (actions: add, list, complete, delete)."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "complete", "delete"],
                "description": "Action to perform: 'add', 'list', 'complete', or 'delete'.",
            },
            "title": {
                "type": "string",
                "description": "Title/description of the todo item (required for 'add').",
            },
            "todo_id": {
                "type": "integer",
                "description": "ID of the todo item (required for 'complete' or 'delete').",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Priority level of the todo item.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, store: Optional[TodoStore] = None):
        self.store = store or TodoStore()
        # 默认会话绑定
        self._current_session_id = "default"

    def set_session_id(self, session_id: str):
        """
        动态绑定当前活跃会话 ID。
        由 AgentRuntime 在每次 run() 调度时动态调用，确保窗口 1 与窗口 2 各自独立。
        """
        self._current_session_id = session_id

    def run(
        self,
        action: str,
        title: Optional[str] = None,
        todo_id: Optional[int] = None,
        priority: str = "normal",
        **kwargs,
    ) -> Dict[str, Any]:
        # 优先使用显式传入的 session_id，否则使用当前上下文绑定的 session_id
        session_id = kwargs.get("session_id", self._current_session_id)

        if action == "add":
            if not title:
                raise ValueError("Parameter 'title' is required when action='add'")
            item = self.store.add(session_id, title=title, priority=priority)
            return {"status": "success", "message": f"Added todo #{item['id']}: '{title}'", "todo": item}

        elif action == "list":
            items = self.store.list(session_id)
            return {"status": "success", "session_id": session_id, "total": len(items), "todos": items}

        elif action == "complete":
            if todo_id is None:
                raise ValueError("Parameter 'todo_id' is required when action='complete'")
            item = self.store.complete(session_id, todo_id=int(todo_id))
            if not item:
                return {"status": "error", "message": f"Todo #{todo_id} not found in session '{session_id}'"}
            return {"status": "success", "message": f"Todo #{todo_id} marked as completed", "todo": item}

        elif action == "delete":
            if todo_id is None:
                raise ValueError("Parameter 'todo_id' is required when action='delete'")
            ok = self.store.delete(session_id, todo_id=int(todo_id))
            if not ok:
                return {"status": "error", "message": f"Todo #{todo_id} not found in session '{session_id}'"}
            return {"status": "success", "message": f"Todo #{todo_id} deleted"}

        else:
            raise ValueError(f"Unknown action '{action}'. Valid actions are add, list, complete, delete.")


class WeatherTool(BaseTool):
    """
    天气查询工具。
    支持查询指定城市（广州、厦门、北京、上海、深圳等）的即时天气与气温。
    """
    name = "weather"
    description = "Query current weather conditions and temperature for a given city."
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Beijing', 'Guangzhou', 'Xiamen', 'Shanghai'.",
            }
        },
        "required": ["city"],
    }

    WEATHER_DATA = {
        "guangzhou": {"weather": "Sunny with light clouds", "temperature": "26°C", "humidity": "65%", "air_quality": "Good"},
        "xiamen": {"weather": "Clear sea breeze", "temperature": "24°C", "humidity": "70%", "air_quality": "Excellent"},
        "beijing": {"weather": "Sunny and crisp", "temperature": "18°C", "humidity": "40%", "air_quality": "Moderate"},
        "shanghai": {"weather": "Partly cloudy", "temperature": "20°C", "humidity": "60%", "air_quality": "Good"},
        "shenzhen": {"weather": "Warm and sunny", "temperature": "27°C", "humidity": "68%", "air_quality": "Good"},
    }

    def run(self, city: str, **kwargs) -> Dict[str, Any]:
        city_key = city.lower().strip()
        data = self.WEATHER_DATA.get(city_key)
        if not data:
            data = {"weather": "Sunny", "temperature": "22°C", "humidity": "55%", "air_quality": "Good"}

        return {"city": city, **data}
