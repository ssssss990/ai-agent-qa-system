"""
LLM 客户端：模型路由

按问题类型分流：
- 常规 FAQ -> 本地 Qwen2.5-7B（vLLM），延迟低、无 API 费用
- 复杂推理 -> DeepSeek-V2.5 API，推理能力强

分流逻辑：根据问题的复杂度判断
- 含"对比""分析""为什么"等推理词 -> 走远程
- 其他走本地
"""

import re
from typing import Optional
from openai import AsyncOpenAI
from config.settings import settings


class LLMClient:
    """LLM 客户端，支持模型路由"""

    def __init__(self):
        self.local_client = AsyncOpenAI(
            base_url=settings.local_model_base_url,
            api_key="vllm",
        )
        self.remote_client = AsyncOpenAI(
            base_url=settings.remote_model_base_url,
            api_key="sk-placeholder",
        )

        self.reasoning_keywords = [
            "对比", "分析", "为什么", "区别", "原理",
            "比较", "优缺点", "怎么实现", "为什么",
        ]

    async def chat(
        self,
        messages: list[dict],
        model_type: str = "generation",
    ) -> str:
        """
        调用 LLM

        Args:
            messages: 对话消息列表
            model_type: "generation" 走本地，"reasoning" 走远程

        Returns:
            LLM 生成的文本
        """
        if model_type == "reasoning":
            return await self._call_remote(messages)
        return await self._call_local(messages)

    async def _call_local(self, messages: list[dict]) -> str:
        """调用本地 vLLM 部署的 Qwen2.5-7B"""
        try:
            response = await self.local_client.chat.completions.create(
                model=settings.local_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Local model failed: {e}, falling back to remote")
            return await self._call_remote(messages)

    async def _call_remote(self, messages: list[dict]) -> str:
        """调用远程 DeepSeek-V2.5 API"""
        response = await self.remote_client.chat.completions.create(
            model=settings.remote_model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def should_use_reasoning(self, query: str) -> bool:
        """判断是否需要推理模型"""
        return any(kw in query for kw in self.reasoning_keywords)
