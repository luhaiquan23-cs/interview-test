"""
DeepSeek LLM 客户端封装
----------------------------------------------------------------------
负责对接 DeepSeek API（通过 OpenAI-compatible SDK 协议通信）。
主要职责:
  1. 安全读取并校验环境变量中的 DEEPSEEK_API_KEY；
  2. 支持 Function Calling 工具定义传入与格式化；
  3. 支持从返回结果中提取模型的思考过程 (Thought/Reasoning Content)；
  4. 支持自适应重试与网络超时防护。
"""

import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from .schema import Message, ToolCall

# 加载 .env 本地环境变量
load_dotenv()


class DeepSeekClient:
    """
    DeepSeek API 客户端封装。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        # 优先读取入参，缺省从环境变量 DEEPSEEK_API_KEY 读取
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Please set it in your .env file or pass it to DeepSeekClient."
            )
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.max_retries = max_retries

        # 初始化底层 OpenAI 兼容客户端
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.6,
    ) -> Tuple[Optional[str], Optional[str], List[ToolCall], str]:
        """
        向 DeepSeek 发送聊天补全请求并解析响应。

        参数:
          - messages: 上下文消息列表
          - tools: 已注册工具的 JSON Schema 定义
          - temperature: 生成随机度

        返回元组:
          - content: 模型直接生成的文本内容（若有）
          - thought: 模型的思考链过程（若支持或输出）
          - tool_calls: 模型请求调用的结构化工具列表
          - finish_reason: 终止原因 ('stop', 'tool_calls', 等)
        """
        # 序列化转换为 OpenAI API 要求的 Dict 格式
        api_messages = [msg.to_openai() for msg in messages]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
        }

        # 仅当注册了工具时才传入 tools 规范
        if tools and len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # 带指数退避的重试机制
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                msg = choice.message

                # 1. 提取思考过程 (Thought / Reasoning Content)
                # 兼容 DeepSeek Reasoner 的 reasoning_content 属性
                thought = getattr(msg, "reasoning_content", None)
                content = msg.content or ""

                # 兼容模型直接在 content 中输出 <think>...</think> 标签的推理模式
                if not thought and "<think>" in content:
                    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                    if match:
                        thought = match.group(1).strip()
                        # 从最终输出文本中剔除思考标签
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                # 2. 提取工具调用 (Tool Calls)
                tool_calls: List[ToolCall] = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(ToolCall.from_openai(tc))

                finish_reason = choice.finish_reason or ("tool_calls" if tool_calls else "stop")
                return content, thought, tool_calls, finish_reason

            except Exception as e:
                # 达到最大重试次数时抛出异常
                if attempt == self.max_retries:
                    raise RuntimeError(f"DeepSeek API call failed after {self.max_retries + 1} attempts: {str(e)}") from e
                time.sleep(1.0 * (attempt + 1))
