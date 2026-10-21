# 2026 年 Agent Harness 架构设计题深度解析

> 本文档针对 5 大核心架构模块各精选一道核心题目，结合生产级 Agent Runtime（如 Claude Code、OpenHands、LangGraph 等工程实践）进行深度剖析与方案设计。

---

## 模块一：Context / Performance
### 题目 1：大模型面对第一轮长窗口或多模态输入时，First Token (TTFT) 会显著变慢（5-10秒）。有什么快速、低成本、用户体验也不差的方案将其稳定压缩到 2 秒？

### 深度解析与架构设计

#### 1. 瓶颈根因分析 (Root Cause)
首字延时高（Time To First Token, TTFT）的本质是 **Prefill 阶段的计算与传输瓶颈**：
- **KV Cache 未命中**：面对上万 Token 的长文档或高清图片输入，推理引擎必须对全量 Prompt 进行完整的矩阵计算，Prefill 耗时与 Context 长度呈二次方或超线性增长；
- **网络传输与预处理开销**：多模态图像/音视频的编码转换耗时。

#### 2. 生产级优化方案组合拳（实测压缩至 2 秒内）

```
[用户长输入 / 多模态]
       │
       ├───> [快速通道 Fast-path] (小模型 / 模板引擎) ───> 立即输出 Acknowledgement 流式占位 (200ms)
       │
       ├───> [Prompt Caching 前缀对齐] ─────────────> 命中云端 KV Cache (Prefill 耗时下降 80%)
       │
       └───> [多模态自适应降采样 + 分块预处理] ─────────> Token 压缩 60% 且保留局部高精特征
```

1. **Prompt 结构规范化与 KV Cache 强对齐 (Prefix Caching / Prompt Cache)**
   - **原理**：现代大模型推理服务（如 DeepSeek、Anthropic、OpenAI）支持 KV Cache 自动复用。只要静态前缀完全一致即可跳过 Prefill 计算；
   - **设计**：将 Prompt 严格切分为 **[绝对静态系统提示] + [静态工具定义] + [长参考文档] + [易变的当前用户指令]**。确保前三者严格对齐内存页（Cache Line，如 1024 Token 块对齐），使首轮命中的缓存比例最大化。实测 Prefill 耗时可直接下降 70%~85%。

2. **双通道流水线（Dual-Stream UX 补偿方案）**
   - **体验优化**：在底层长 Prefill 运算的同时，前端/网关层立即启动本地微型意图模型（或规则引擎），在 **150ms 内** 流式吐出语义化的确认帧（例如：“*正在解析 82 页财报文档与图表，已定位核心章节...*”）；
   - **价值**：将用户心理等待阈值重置，同时让用户感知到系统正在实质性处理。

3. **多模态与长文档动态金字塔降采样 (Dynamic Pyramidal Compression)**
   - 对于图片/扫描件：不盲目传入 4K 原图，第一轮仅传入 512x512 低分辨率 Global View（约几百 Token）先生成初始规划；当模型决策需要细节时，再通过工具二次裁剪局部 High-res Patch。
   - 对于长文档：在网关层做极速的 BM25/BGE-m3 局部 Chunk 过滤，将首轮输入从 50k Token 裁剪至 4k 高相关 Token，剩余内容转为 Agent 工具检索模式。

---

## 模块二：Memory
### 题目 1：和聊天 Agent 熟悉半个月后，用户问了一个以前问过的问题。Agent 如何做 Memory 召回更合理？

### 深度解析与架构设计

#### 1. 核心困境
用户半个月后的提问（如“*上次我选的那套方案最终版发我下*”或“*我的车牌号是多少来着*”）：
- **时序跨度长**：上下文窗口早已翻篇数十万 Token；
- **表达模糊/歧义**：用户使用了代词（“上次”、“那套方案”）；
- **召回污染风险**：若直接全局向量搜索，极易把半个月内的相似无关废话捞出来塞进 Context，造成注意力漂移与幻觉。

#### 2. 分层召回架构 (Hierarchical Memory Architecture)

```
用户输入: "上次我选的那套方案最终版发我下"
                   │
                   ▼
       [第一阶段：意图识别与查询重写]
       - 检测到隐式回溯指代 ("上次", "最终版")
       - 生成检索 Query: [语义: "方案 最终版", 时序过滤: [now-30d to now]]
                   │
                   ▼
       [第二阶段：混合检索与时序衰减评分 (Hybrid Scoring)]
       Score = 0.5 * Dense_Similarity 
             + 0.3 * Sparse_BM25 
             + 0.2 * Temporal_Decay(t) * Priority_Weight
                   │
                   ▼
       [第三阶段：记忆提炼与防污染放置]
       - Top-K 记忆片段经由轻量 Rerank 模型重排
       - 格式化后注入 [Relevant Past Recollections] 区块
```

#### 3. 关键实现要点

1. **时序衰减与事件归纳 (Episodic Memory)**：
   - 原始半个月的聊天记录绝不能存为裸文本，而应在每日会话结束或闲时离线提炼为 **事件单元 (Episodic Node)**：`{timestamp, topic, key_decisions, entities}`；
   - 召回计算公式中加入时间权重衰减（半衰期模型）与“决策标签（Decision/Commitment）”加权，优先召回产生明确结论的事件。

2. **召回放置方式**：
   - 放置在 System Prompt 之后的单独区块：
     ```markdown
     <episodic_memory source="2026-10-05">
     - 用户在方案 A 与方案 B 中最终敲定了方案 A (极简版)，并确认了包含预算清单。
     </episodic_memory>
     ```
   - 严禁将检索到的海量原始对话无序平铺在最近的 User 消息前面。

---

## 模块三：Task
### 题目 1：对于长程任务（Long-Horizon Tasks），大模型执行一段时间可能会忘掉目标（Goal Drift）。你知道哪些解决方案？各自有什么优缺点？

### 深度解析与方案对比

长程任务中大模型失忆的根本原因在于：**有限的注意力跨度（Attention Degradation）**、**自回归过程中的累积误差**以及**工具执行结果噪声对初始系统提示的稀释**。

| 解决方案 | 核心机制 | 优点 | 缺点 / 代价 |
|---------|---------|------|------------|
| **1. 状态外置与 Scratchpad (Plan-and-Solve)** | 在 Context 外维护全局状态机与 Todo 清单。每一步执行前强制将当前 Goal、Completed Steps、Next Steps 固化在上下文最醒目位置。 | • 确定性强，不易丢失主线；<br>• 可读性极高，易于调试追溯。 | • 增加了 Prompt 的冗余度；<br>• 若模型中途错误更新了 Scratchpad，可能引发连锁误判。 |
| **2. 双层分级 Agent 架构 (Supervisor - Worker)** | 上层 Supervisor 只负责任务分解、全局状态管理与目标把控；下层 Worker 专注于单步原子工具调用，执行完毕后向 Supervisor 汇报。 | • 彻底解耦“全局规划”与“局部执行”；<br>• Worker 的 Context 永远干净，极大降低 Token 消耗。 | • 系统通信开销增大；<br>• 调用链路延迟翻倍；<br>• 实现与并发协调复杂度较高。 |
| **3. 恒定注意力再锚定 (System Invariant Anchor)** | 在多轮循环中，不把初始任务当做普通的早期 User 消息，而是作为不可侵犯的 **System Invariant**，在每一轮 API 请求中固定置于 Prompt 末尾（Recency Bias 利用）。 | • 实现简单，零改造成本；<br>• 有效对抗大模型的“中途失忆”效应。 | • 占用固定的 Prompt 空间；<br>• 灵活性较弱，无法动态应对用户中途变更目标的情形。 |
| **4. 状态核验与 Critic 反思机制 (Reflection)** | 引入独立 Critic 节点，每执行 $N$ 步或关键操作后，评估“当前执行进度与初始目标的偏差度”，一旦发散则强制纠偏。 | • 自愈能力强；<br>• 显著提高复杂代码重构或长流程自动化成功率。 | • API 成本增加 30%~50%；<br>• 对 Critic 模型的推理与反思能力要求极高。 |

**工程最佳实践推荐**：
在轻量场景采用 **方案 1 + 方案 3**（Scratchpad 动态状态维护 + Recency Prompt 锚定）；在重量级生产场景（如类似 SWE-bench 的代码工程 Agent）采用 **方案 2（双层架构）**。

---

## 模块四：Tool / Session Runtime
### 题目 2：如果 Session 状态为 Busy，此时用户又发来新消息，或者异步工具完成事件也到达，Runtime 应该如何处理？

### 深度解析与架构设计

#### 1. 核心挑战
在单线程传统的线性 Agent 中，若正在执行一个耗时 15 秒的代码跑测工具，此时：
- 用户发来：“*不对，刚才参数写错了，停一下改成 8080*”；
- 或者先前的异步分析任务正好返回了结果。
如果简单粗暴地阻塞丢弃，或者直接强行并发写入 Context，会导致**上下文时序混乱、模型幻觉、幻影工具调用（Ghost Tool Calls）**。

#### 2. 现代 Runtime 的事件驱动与 Actor 架构

```
               [新用户消息]          [异步工具回调事件]
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Session Event Queue  │  (优先级事件队列)
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Runtime Coordinator  │
                    │  (State: BUSY)        │
                    └───────────┬───────────┘
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
  [策略 1: 抢占打断]       [策略 2: 合并排队]     [策略 3: 静默写入]
  用户发明确打断词          常规追加指令           纯通知类异步结果
  (AbortController)      (Wait current step)   (Next step observed)
```

#### 3. 具体处理策略矩阵

1. **用户发来新消息：**
   - **意图抢占检测 (Preemption / Cancellation)**：
     - 如果用户输入包含打断/重置意图（如“停止”、“重新来”、“改一下”）：Runtime 立即向当前执行中的工具发送 `AbortSignal`（如杀死子进程或中断 HTTP 请求），将当前正在推理或执行的 Step 标记为 `Interrupted`，清空未消费队列，立刻响应新消息。
   - **排队合并 (Message Queue & Batching)**：
     - 如果是非打断类补充消息（如“*顺便帮我也查下明天的数据*”）：消息暂存入该 Session 的 Inbox。等当前原子步（Atomic Step）完成后，将 Inbox 中的所有新消息合并为单一复合消息，作为下一步的输入。

2. **异步工具完成事件到达：**
   - **原子步隔离原则 (Turn-boundary Injection)**：
     - 异步结果绝不打断正在进行的 LLM Token 流式吐字；
     - 异步事件进入 Session 的 `PendingObservations` 缓冲区。当当前推理步结束（LLM 停词或当前同步工具返回），Runtime 在开启下一步循环前，将缓冲区中的结果统一封装为 `tool` 消息追加至 Context，供模型在下一轮感知。

3. **版本控制与乐观锁 (Context Versioning)**：
   - 每次上下文写入均带有单调递增的 `context_version`；
   - 任何工具执行返回时必须核对版本。若 Session 已被用户打断重置，过期的异步结果直接静默丢弃（Zombie Event Dropping）。

---

## 模块五：Agent Runtime 架构对比
### 题目 1：Claude Code 的工具输出方式和国内 GLM / 豆包等 OpenAI-compatible Function Calling 有什么不同？各自这样设计的优缺点是什么？

### 深度解析与架构对比

#### 1. 两种设计哲学的本质分野

- **OpenAI-Compatible Function Calling（GLM / 豆包 / 传统 Agent）**：
  - **规范化的结构化 RPC 协议**：严格要求开发者提前注册定义好的 JSON Schema；模型输出严格结构化的 JSON 字段；Runtime 校验解析后调用对应 Python/RPC 函数。
- **Claude Code 风格（基于 Bash / File System 的原生文本交互）**：
  - **基于操作系统原生环境的自然控制**：核心仅提供极少数原生系统工具（如 `Bash`, `FileEdit`, `Glob`）；模型就像一个真实的工程师，通过命令行标准输入输出（stdin/stdout/stderr）与系统对话。

#### 2. 全方位维度对比表

| 对比维度 | OpenAI-Compatible Function Calling | Claude Code (原生 Bash/环境式) |
|---------|-----------------------------------|-------------------------------|
| **交互协议** | JSON-RPC 格式，强类型，严格 Schema 校验 | 终端命令行字符串（Shell Command）与文件流 |
| **工具扩展性** | 每加一个新能力，必须编写并注入一套完整的 JSON Schema，工具多了会导致 Context 膨胀 | **无限扩展**。无需额外注册，模型可通过 Bash 自行调用 `grep`, `git`, `curl`, `jq`, `pytest` 等系统中已有的一切工具 |
| **Token 消耗** | 预置 20 个复杂工具可能占用数千 Token 的系统提示空间 | 工具定义极轻量（仅需声明“你有 Bash 权限”），几乎不占预置 Context |
| **容错与恢复** | 参数类型不符合 Schema 会直接阻断报错抛异常，难以动态组合 | 报错即标准 `stderr`，模型根据终端错误日志自纠错能力极强，支持管道组合（`cat file \| grep err`） |
| **安全性与沙箱** | 高，参数受限，只能调用已暴露的受控白名单 API | 极度依赖严格的 OS 容器/沙箱隔离（必须在 Docker/MicroVM 内运行），否则极易造成不可逆损坏 |
| **生态与集成** | 简单，与现有企业微服务/RESTful API 契合度极高 | 复杂，需要完整的执行容器、文件系统与伪终端（PTY）支撑 |

#### 3. 架构设计总结
- **Function Calling** 适用于**业务受限型系统**（如客服、订单查询、企业审批、移动端 App 控制），强类型、确定性高、安全性可控；
- **Claude Code 原生环境交互** 是面向**软件工程、数据科学和高阶自主 Agent** 的最终形态，它将大模型从“API 调度员”真正转变成了“数字工作者”。

