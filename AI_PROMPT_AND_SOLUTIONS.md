# AI Prompt 与问题解决记录

> 本记录详细归档在基于 DeepSeek API 从零实现 Minimal Agent 过程中的关键 AI Prompt、架构选型思考、踩坑排查过程与工程解决方案。

---

## 一、AI Prompt 交互记录

### 1. 架构设计与 4 步循环生成 Prompt

**Prompt 输入：**
> “请帮我设计一个纯 Python 实现的最小可用 Agent Runtime，严禁使用任何第三方 Agent 框架（如 LangGraph/OpenHands/CrewAI）。
> 核心要求：
> 1. 严格实现 4-Step Loop：接收输入 -> 判断直接回复或调工具 -> 执行工具 -> 判断继续循环或返回结果；
> 2. 支持工具注册机制（包含 Name、Description、JSON Schema）；
> 3. 支持多 Session 独立隔离（如窗口 1 查天气加待办，窗口 2 写周报加待办）；
> 4. 支持 Context 上下文管理（最大轮次与基础压缩）；
> 5. 异常处理与执行 Trace 日志。请提供清晰解耦的模块划分。”

**AI 输出价值：**
- 明确了模块划分为 `schema.py`（数据载荷）、`llm.py`（DeepSeek 接口层）、`tools/`（工具与注册中心）、`context.py`（上下文与滑动窗口）、`session.py`（会话隔离）以及 `runtime.py`（核心调度器）。

---

### 2. 工具安全实现 Prompt（防止 RCE 代码执行）

**Prompt 输入：**
> “在实现 Calculator 工具时，很多 Agent 会直接使用 `eval()`，但在生产环境这存在任意代码执行漏洞（如 `__import__('os').system()`）。
> 请为我设计一个基于 Python `ast`（抽象语法树）解析的安全计算器，只允许白名单中的数学四则运算、幂运算以及常用数学函数（sqrt, sin, log, round），彻底拒绝危险语法节点。”

**AI 输出价值：**
- 构建了 `SafeCalculator` 类，通过遍历 AST 节点（`ast.BinOp`, `ast.UnaryOp`, `ast.Call`, `ast.Constant`）进行白名单运算符映射，实现了安全的数学表达式解析。

---

## 二、关键问题排查与工程解决方案 (Troubleshooting)

### 问题 1：DeepSeek API 报 422 校验失败 (`invalid type: map, expected a string`)

#### 1. 现象描述
在首次运行端到端真实 API 测试时，纯对话用例通过，但在工具调用（Calculator/Todo）完成第二轮请求时，DeepSeek API 报错中断：
```
Error code: 422 - {
  'error': {
    'message': 'Failed to deserialize the JSON body into the target type: messages[2]: invalid type: map, expected a string at line 1 column 828',
    'type': 'invalid_request_error'
  }
}
```

#### 2. 原因分析
检查传入 DeepSeek 的消息体结构：
- `messages[0]` 为 `system`
- `messages[1]` 为 `user`
- `messages[2]` 为 `assistant`，其中带有 `tool_calls`
- 在我们的初始实现中，给 `function.arguments` 直接传递了 Python `dict`（即 JSON 中的 Map 对象）：
  ```json
  "tool_calls": [{
    "id": "call_123",
    "type": "function",
    "function": {
      "name": "calculator",
      "arguments": {"expression": "384 * 25"}  // <-- 错误：这是 Map
    }
  }]
  ```
- 根据 OpenAI / DeepSeek API 的严格规范，`assistant` 消息中返回给 API 的 `tool_calls[i].function.arguments` **必须是 JSON 序列化后的字符串**，而不是原生对象！

#### 3. 解决方案
在 `AgentRuntime.run()` 中装配 assistant 历史消息时，对 `tc.arguments` 进行序列化包装：
```python
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
```
修复后重新运行测试，E2E 测试全部顺利通过。

---

### 问题 2：模块循环依赖导致导入失败 (Circular Import)

#### 1. 现象描述
在运行 pytest 收集测试用例时报错：
```
ImportError: cannot import name 'ToolRegistry' from partially initialized module 'agent.tools.base' (most likely due to a circular import)
```

#### 2. 原因分析
依赖拓扑图如下：
1. `tests/test_tools.py` 导入 `agent.tools.builtin`；
2. `agent.tools.base` 导入 `from agent.core.schema import ToolCall`；
3. Python 加载 `agent.core` 时触发了 `agent/core/__init__.py`；
4. `agent/core/__init__.py` 中原本直接导入了 `from .runtime import AgentRuntime`；
5. `AgentRuntime` 又反向导入了 `from agent.tools.base import ToolRegistry`；
6. 此时 `agent.tools.base` 尚未完成模块初始化，形成了死锁。

#### 3. 解决方案
- 解耦模块初始化：`schema.py` 作为最底层的数据载荷定义，不依赖任何上层业务逻辑；
- 在 `agent/core/__init__.py` 中移除对 `AgentRuntime` 的急切加载（Eager import），改为显式引入或分层引入；
- 并在 `pytest.ini` 中指定 `pythonpath = .`，确保导入路径规范统一。

---

### 问题 3：多 Session 窗口下 Todo 状态的污染问题

#### 1. 现象描述
用户要求实现：
- 窗口 1：查天气 + 记待办
- 窗口 2：写周报 + 记待办
- 两个窗口是独立 session，随时切换彼此不影响。
如果 Todo 工具内部使用全局单例存储，窗口 1 记的待办会被窗口 2 的 Agent 读取或篡改。

#### 2. 解决方案
- 设计 `TodoStore`，底层存储结构使用分桶字典：`Dict[session_id, List[TodoItem]]`；
- 在 `TodoTool` 中添加 `set_session_id(session_id)` 接口；
- 在 `AgentRuntime.run()` 开始时，根据当前执行的 `session_id` 动态将当前上下文会话注入工具：
  ```python
  todo_tool = self.tools.get_tool("todo")
  if isinstance(todo_tool, TodoTool):
      todo_tool.set_session_id(session_id)
  ```
- 这样即使底层共享同一个 Todo 管理器实例，各窗口的数据在存储层也天然物理隔离。

---

### 问题 4：长文本下 Context 的平滑压缩策略

#### 1. 考量与权衡
当单会话对话达到 10~20 轮以上时，Context 会逐渐逼近或消耗大量 Token，且容易引起模型注意力漂移。
如果直接粗暴截断（如只留最近 2 轮），模型会彻底丢失之前的背景信息；如果每轮都调用 LLM 做一次摘要，会造成额外的 API 延迟和成本。

#### 2. 解决方案（轻量双缓冲压缩）
- 采用**滑动窗口 + 结构化增量压缩**：
  1. 设置压缩阈值 `compression_threshold = 12`；
  2. 触发时将早期消息以结构化文本格式（`User asked... / Agent called... / Tool returned...`）快速提取提炼，合并到系统提示词下的 `[Summary of Prior Conversation]` 区块；
  3. 完整保留最近的 6 条对话。
- 这样既保证了不需要额外发起 LLM 摘要请求（快速、零成本），又保证了近期对话的细节精度和长期历史的语义连贯。

