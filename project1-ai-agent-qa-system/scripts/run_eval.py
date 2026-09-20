"""
评测脚本

用法：
python scripts/run_eval.py --dataset data/eval/eval_dataset.json

输出各维度指标，并与上次结果对比（如果存在历史记录）
"""

import asyncio
import argparse
import json

from src.evaluation.evaluator import RAGEvaluator
from src.agent.react_agent import ReActAgent
from src.tools.rag_tool import RAGSearchTool
from src.utils.llm_client import LLMClient


async def run_evaluation(dataset_path: str):
    evaluator = RAGEvaluator()
    evaluator.load_dataset(dataset_path)

    llm = LLMClient()
    rag_tool = RAGSearchTool()
    agent = ReActAgent(llm=llm, tools={"rag_search": rag_tool})

    async def agent_fn(query: str) -> dict:
        return await agent.run(query)

    results = await evaluator.run_full_eval(agent_fn)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)

    with open("data/eval/last_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to data/eval/last_result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/eval_dataset.json")
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.dataset))
