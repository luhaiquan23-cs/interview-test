"""
命令行交互控制台 (CLI Runner)
----------------------------------------------------------------------
基于 Rich 终端库构建的交互式 Agent 对话与调试控制台。
功能特色:
  1. 多窗口会话切换 (/session <id>)：模拟用户在不同窗口切换（如窗口 1 查天气记待办，窗口 2 写周报记待办）；
  2. 会话状态一览 (/sessions)：查看所有处于存活状态的会话及其消息量；
  3. 待办清单速览 (/todos)：直接以表格形式查看当前会话绑定的 Todo 项；
  4. 链路追踪开关 (/trace)：控制是否实时打印模型思考过程、工具调用与耗时；
  5. 记忆注入 (/memory <txt>)：测试用户长期记忆注入与系统提示词联动；
  6. 上下文重置 (/reset)：清空当前窗口上下文。
"""

import sys
import os
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from agent.core.runtime import AgentRuntime
from agent.core.schema import AgentStep
from agent.tools.builtin import TodoTool

console = Console()


def format_step_trace(step: AgentStep):
    """
    格式化并高亮渲染单步执行快照 (AgentStep) 至终端。
    
    展示信息包含:
      - 步骤序号与动作类型（直接回复 direct_response 或工具调用 tool_call）
      - 步骤耗时（毫秒）
      - 模型思考推理过程（Thought/Reasoning）
      - 发起的工具名称及输入实参
      - 工具执行成功/失败状态与输出文本
    """
    step_type_color = "cyan" if step.action == "tool_call" else "green"
    header = f"[bold {step_type_color}]Step {step.step_index} ({step.action})[/bold {step_type_color}] - {step.duration_ms:.1f}ms"

    body_parts = []
    # 渲染模型思考链（如果有）
    if step.thought:
        body_parts.append(f"[bold yellow]Thought:[/bold yellow] {step.thought}")

    # 渲染发起的工具调用
    if step.tool_calls:
        for tc in step.tool_calls:
            body_parts.append(f"[bold magenta]Tool Call:[/bold magenta] [cyan]{tc.name}[/cyan]({tc.arguments})")

    # 渲染工具执行返回值
    if step.tool_results:
        for res in step.tool_results:
            status_text = "[green]✓ Success[/green]" if res.success else "[red]✗ Failed[/red]"
            body_parts.append(f"[bold blue]Tool Result ({res.name}) [{status_text}]:[/bold blue]\n{res.content}")

    # 渲染最终文本回复
    if step.response_content and step.action == "direct_response":
        body_parts.append(f"[bold green]Response Content:[/bold green] {step.response_content}")

    panel = Panel("\n".join(body_parts), title=header, border_style=step_type_color)
    console.print(panel)


def main():
    """
    CLI 交互主循环入口。
    初始化 Agent 运行时，处理用户控制台输入与特殊斜杠指令调度。
    """
    console.print(Panel.fit(
        "[bold cyan]Zero-Framework Minimal Agent Runtime[/bold cyan]\n"
        "[dim]Powered by DeepSeek API & Custom 4-Step Event Loop[/dim]\n\n"
        "Commands:\n"
        "  [yellow]/session <id>[/yellow] - Switch or create a session window\n"
        "  [yellow]/sessions[/yellow]     - List all open sessions\n"
        "  [yellow]/todos[/yellow]        - List todos in current session\n"
        "  [yellow]/trace[/yellow]        - Toggle step execution tracing\n"
        "  [yellow]/memory <txt>[/yellow] - Inject long-term memory\n"
        "  [yellow]/reset[/yellow]        - Clear context in current session\n"
        "  [yellow]/exit[/yellow]         - Exit program",
        border_style="cyan"
    ))

    # 初始化配置：默认开启单步执行 Trace 渲染
    verbose_trace = True

    def on_step(step: AgentStep):
        """每完成一步事件循环时的实时回调"""
        if verbose_trace:
            format_step_trace(step)

    try:
        # 初始化 AgentRuntime
        runtime = AgentRuntime(trace_callback=on_step)
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        sys.exit(1)

    # 默认进入窗口 1 (window_1)
    current_session = "window_1"
    console.print(f"\n[bold green]Connected to session:[/bold green] [bold cyan]{current_session}[/bold cyan]\n")

    # 终端交互式死循环
    while True:
        try:
            prompt = console.input(f"[bold yellow]({current_session}) User > [/bold yellow]").strip()
            if not prompt:
                continue

            # ---------------- 指令路由分支 ----------------
            # 1. 退出程序
            if prompt.lower() in ("exit", "quit", "/exit"):
                console.print("[dim]Goodbye![/dim]")
                break

            # 2. 切换或新建窗口会话 (/session <id>)
            elif prompt.startswith("/session "):
                new_session = prompt.split(" ", 1)[1].strip()
                if new_session:
                    current_session = new_session
                    console.print(f"[bold green]Switched to session window:[/bold green] [bold cyan]{current_session}[/bold cyan]")
                continue

            # 3. 列出所有活动会话 (/sessions)
            elif prompt == "/sessions":
                all_sessions = runtime.sessions.list_sessions()
                table = Table(title="Active Session Windows")
                table.add_column("Session ID", style="cyan")
                table.add_column("Messages", style="green")
                table.add_column("Status", style="yellow")
                for sid in all_sessions:
                    sess = runtime.sessions.get(sid)
                    cnt = len(sess.context.get_messages()) if sess else 0
                    cur = "Active *" if sid == current_session else ""
                    table.add_row(sid, str(cnt), cur)
                console.print(table)
                continue

            # 4. 快速查看当前窗口绑定的 Todo 待办列表 (/todos)
            elif prompt == "/todos":
                todo_tool = runtime.tools.get_tool("todo")
                if isinstance(todo_tool, TodoTool):
                    todos = todo_tool.store.list(current_session)
                    table = Table(title=f"Todos for [{current_session}]")
                    table.add_column("ID", style="dim")
                    table.add_column("Title", style="white")
                    table.add_column("Priority", style="yellow")
                    table.add_column("Done", style="green")
                    for t in todos:
                        done_mark = "[green]✓[/green]" if t["completed"] else "[dim]o[/dim]"
                        table.add_row(str(t["id"]), t["title"], t["priority"], done_mark)
                    console.print(table)
                continue

            # 5. 开关步进 Trace 日志显示 (/trace)
            elif prompt == "/trace":
                verbose_trace = not verbose_trace
                status = "enabled" if verbose_trace else "disabled"
                console.print(f"[dim]Step trace {status}.[/dim]")
                continue

            # 6. 为当前会话注入长期记忆 (/memory <txt>)
            elif prompt.startswith("/memory "):
                mem_text = prompt.split(" ", 1)[1].strip()
                sess = runtime.sessions.get_or_create(current_session)
                sess.context.add_memory(mem_text)
                console.print(f"[green]Added memory to session {current_session}:[/green] '{mem_text}'")
                continue

            # 7. 重置当前窗口上下文 (/reset)
            elif prompt == "/reset":
                sess = runtime.sessions.get(current_session)
                if sess:
                    sess.context.reset()
                console.print(f"[yellow]Context reset for session {current_session}.[/yellow]")
                continue

            # ---------------- 触发 Agent 核心执行循环 ----------------
            console.print(f"[dim]Agent thinking...[/dim]")
            answer, trace = runtime.run(prompt, session_id=current_session)

            # 打印最终输出面板
            console.print(Panel(
                answer,
                title=f"[bold green]Agent Answer ({trace.total_duration_ms:.1f}ms)[/bold green]",
                border_style="green"
            ))

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session closed.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
