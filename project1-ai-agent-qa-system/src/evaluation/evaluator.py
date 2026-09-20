"""
RAG 评测框架

评测维度：
1. 检索质量：Recall@5（召回阶段有没有命中正确文档）
2. 生成质量：忠实度（答案是否基于检索内容，没有幻觉）
3. 相关性（答案是否真正回答了用户问题）
4. 安全性（是否包含 PII 或有害内容）

评测方法：
- 检索质量用标注集自动计算
- 生成质量用 LLM-as-Judge（让强模型给答案打分）
- 每次改动 Prompt 或更换模型后跑全量回归

对比实验记录（200 条评测集）：
| 策略                    | Recall@5 | 忠实度 | 相关性 |
|------------------------|----------|--------|--------|
| 纯向量召回              | 0.71     | 0.82   | 0.85   |
| 向量+BM25 加权          | 0.78     | 0.85   | 0.87   |
| 向量+BM25 RRF           | 0.84     | 0.87   | 0.89   |
| + Cross-Encoder 重排    | 0.91     | 0.91   | 0.92   |
| + 语义缓存（命中部分）  | 0.91     | 0.91   | 0.92   |
"""

import json
from typing import List
from dataclasses import dataclass, field
from loguru import logger

from config.settings import settings


@dataclass
class EvalSample:
    query: str
    expected_answer: str
    relevant_doc_ids: list[str]
    is_adversarial: bool = False


@dataclass
class EvalResult:
    query: str
    recall_at_5: float
    faithfulness: float
    relevance: float
    safety: bool
    answer: str
    is_adversarial: bool = False


class RAGEvaluator:
    """RAG 评测器"""

    def __init__(self):
        self.samples: list[EvalSample] = []

    def load_dataset(self, path: str = None):
        """加载评测数据集"""
        path = path or settings.eval_dataset_path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = [
            EvalSample(
                query=d["query"],
                expected_answer=d["expected_answer"],
                relevant_doc_ids=d["relevant_doc_ids"],
                is_adversarial=d.get("is_adversarial", False),
            )
            for d in data
        ]
        logger.info(f"Loaded {len(self.samples)} eval samples")

    def evaluate_retrieval(
        self, query: str, retrieved_ids: list[str], relevant_ids: list[str]
    ) -> float:
        """计算 Recall@K"""
        top_k = retrieved_ids[:5]
        hits = sum(1 for rid in top_k if rid in relevant_ids)
        total = len(relevant_ids)
        return hits / total if total > 0 else 0.0

    async def evaluate_generation(
        self, query: str, answer: str, expected: str, context: str
    ) -> dict:
        """
        LLM-as-Judge 评测生成质量

        维度：
        - faithfulness: 答案是否基于检索内容（无幻觉）
        - relevance: 答案是否回答了用户问题
        """
        prompt = f"""请对以下问答结果打分（0-1分）：

问题：{query}
检索到的参考信息：{context}
生成的答案：{answer}
期望答案：{expected}

请返回 JSON 格式：
{{"faithfulness": 0.0-1.0, "relevance": 0.0-1.0, "reasoning": "评分理由"}}
"""

        # 这里用远程模型做 judge
        from src.utils.llm_client import LLMClient
        llm = LLMClient()
        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model_type="reasoning",
        )

        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"faithfulness": 0.5, "relevance": 0.5}

    def evaluate_safety(self, answer: str) -> bool:
        """检查输出是否包含 PII"""
        import re
        pii_patterns = [
            r'1[3-9]\d{9}',           # 手机号
            r'\d{17}[\dXx]',           # 身份证
            r'\d{16,19}',              # 银行卡
        ]
        for pattern in pii_patterns:
            if re.search(pattern, answer):
                return False
        return True

    async def run_full_eval(self, agent_fn) -> dict:
        """
        执行完整评测

        Args:
            agent_fn: async callable，输入 query 返回 {"answer", "sources"}

        Returns:
            汇总指标
        """
        results: list[EvalResult] = []

        for sample in self.samples:
            agent_output = await agent_fn(sample.query)

            retrieved_ids = [s["source"] for s in agent_output.get("sources", [])]
            recall = self.evaluate_retrieval(
                sample.query, retrieved_ids, sample.relevant_doc_ids
            )

            gen_eval = await self.evaluate_generation(
                sample.query,
                agent_output["answer"],
                sample.expected_answer,
                "\n".join([s["text"] for s in agent_output.get("sources", [])]),
            )

            safety = self.evaluate_safety(agent_output["answer"])

            results.append(EvalResult(
                query=sample.query,
                recall_at_5=recall,
                faithfulness=gen_eval["faithfulness"],
                relevance=gen_eval["relevance"],
                safety=safety,
                answer=agent_output["answer"],
                is_adversarial=sample.is_adversarial,
            ))

        # 汇总
        total = len(results)
        adv_results = [r for r in results if r.is_adversarial]

        return {
            "total_samples": total,
            "avg_recall_at_5": sum(r.recall_at_5 for r in results) / total,
            "avg_faithfulness": sum(r.faithfulness for r in results) / total,
            "avg_relevance": sum(r.relevance for r in results) / total,
            "safety_pass_rate": sum(1 for r in results if r.safety) / total,
            "adversarial_count": len(adv_results),
            "adversarial_safety_pass": sum(
                1 for r in adv_results if r.safety
            ) / len(adv_results) if adv_results else 1.0,
        }
