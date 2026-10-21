# 面试笔试题 —— 任务拆解与完成清单

> 基于 `interview_homework.md`，使用 **DeepSeek API** 实现。

---

## 一、Vibe Coding 题：从零实现最小可用 Agent

### 1. 核心约束

- [x] **禁止使用**现有 Agent 框架（LangGraph / OpenHands / OpenClaw）
- [x] 允许使用 AI 工具辅助开发，但 **Agent Runtime 必须自行实现**
- [x] 使用 **DeepSeek API** 作为真实 LLM 后端（配置于 `.env`，已做 `.gitignore` 防泄露保护）

---

### 2. Agent 基本循环（Loop）

- [x] **Step 1** — 接收用户输入（`session.context.add_user_message`）
- [x] **Step 2** — LLM 判断：直接回复 or 调用工具（解析 Thought、Tool Calls 或 Final Answer）
- [x] **Step 3** — 执行工具调用（`ToolRegistry.execute`，异常捕获与自愈）
- [x] **Step 4** — 根据工具结果判断：继续 Loop or 返回最终结果给用户（防死循环最大步数保护）

---

### 3. 工具系统

#### 3.1 工具注册机制

- [x] 每个工具包含：**名称、描述、参数 Schema**（JSON Schema 规范）
- [x] LLM 基于 Schema 自主决策调用哪个工具
- [x] 实现 LLM 输出解析逻辑：提取 **思考过程**、**工具调用**、**最终答案**

#### 3.2 必须实现的工具（已实现 4 个）

| 工具 | 说明 | 状态 |
|------|------|------|
| `calculator` | 数学计算工具（基于 AST 语法树安全解析，杜绝 eval 注入漏洞） | [x] |
| `search` | 搜索工具（内置模拟事实知识库） | [x] |
| `todo` | 待办事项管理（**必须实现**，增删查改，物理绑定 Session 隔离） | [x] |
| `weather` | 天气查询（可自定义扩展） | [x] |

---

### 4. Session 管理

- [x] 支持多 Session 并存，彼此独立
  - 窗口 1（查天气 + 记待办） vs 窗口 2（写周报 + 记待办）
- [x] 每个 Session 拥有独立的上下文与状态
- [x] 用户可随时切回某个 Session 继续聊天（支持 `/session <id>` 随时切换）

---

### 5. Context 有效管理

- [x] **最大轮次限制**（配置 `max_turns`）
- [x] **对话记忆**：持续对话能记住之前的状态
- [x] **追问支持**
  - [x] 纯对话追问
  - [x] 带工具调用的追问
- [x] **Context 内容选择**：保留用户输入、模型思考过程、Tool Calls、Tool Observation
- [x] **基础 Context 压缩**（滑动窗口 + 增量结构化提取，超出阈值自动压缩至系统提示词摘要区块）

---

### 6. 额外要求

- [x] **基本异常处理**（API 调用失败自动重试、未注册工具异常反馈、语法错误自愈）
- [x] **工具调用 Trace / 执行日志**（完整记录每一步的执行时间、输入、输出、状态，并在终端渲染）

---

### 7. 测试用例

- [x] 构建测试用例，覆盖以上所有功能点（共 **23 个测试用例，100% 全部通过**）：
  - [x] 纯对话（不需要工具）
  - [x] 单次工具调用（Calculator）
  - [x] 多次工具调用（Loop 多轮闭环）
  - [x] 追问（纯对话 + 带工具）
  - [x] 多 Session 独立性（窗口 1 与窗口 2 上下文与 Todo 存储双重隔离）
  - [x] Context 压缩触发（滑动窗口结构化 Compaction）
  - [x] 异常情况处理（危险代码拦截、未注册工具报错自纠错）
  - [x] Todo 工具的增删查改（CRUD）
  - [x] 真实 DeepSeek API 端到端集成测试（E2E Tests）

---

### 8. 提交内容

- [x] **代码**：模块化解耦，已做好 Git 仓库初始化与安全 `.gitignore`
- [x] **README.md**：已完整编写
  - [x] 运行方式（依赖安装、`.env` 配置、CLI 交互说明）
  - [x] 系统架构与 4 步循环 Mermaid 图
  - [x] Memory 的召回时机与放置方式说明
- [x] **AI Prompt 与问题解决记录**（`AI_PROMPT_AND_SOLUTIONS.md`）
- [x] **架构设计题深度解答**（`ARCH_DESIGN.md`，5 大模块各精选一题深度回答）

---

## 二、架构设计题（5 个模块回答完成）

详见 [`ARCH_DESIGN.md`](file:///d:/interview_test/ARCH_DESIGN.md)：
- **模块一：Context / Performance** -> 题 1：First Token 首字延时从 5~10 秒压缩至 2 秒内深度方案（Prompt Caching 对齐 + 双通道 UX 占位 + 多模态动态金字塔压缩）
- **模块二：Memory** -> 题 1：半个月后问以前问过的问题，分层召回与防污染放置架构（时序衰减 + 混合检索 + 系统提示词专区放置）
- **模块三：Task** -> 题 1：长程任务大模型忘掉目标（Goal Drift）的 4 类解决方案及优缺点深度对比（Scratchpad 状态外置、分级 Supervisor-Worker 等）
- **模块四：Tool / Session Runtime** -> 题 2：Session 为 Busy 时收到新消息与异步事件的调度方案（抢占式打断、排队合并、原子步隔离、版本乐观锁）
- **模块五：Agent Runtime 架构对比** -> 题 1：Claude Code 环境原生交互 vs OpenAI-compatible Function Calling 深度剖析与架构权衡
