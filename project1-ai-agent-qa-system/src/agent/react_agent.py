"""
ReAct Agent 状态机

核心设计：
- Thought -> Action -> Observation 循环，最多 8 步
- 每步状态带版本号和快照，异常时回滚到上一稳定态
- 重复工具调用检测：如果连续两步调用相同工具 + 相同参数，终止并返回兜底话术
- 工具失败三级处理：重试 1 次 -> 本地纠错 -> 兜底话术

为什么用 ReAct 而非纯 RAG：
- 纯 RAG 只能做单轮问答，无法处理需要多步推理的问题
- 比如 "对比 A 产品和 B 产品的保修政策" 需要先检索 A 再检索 B 再综合
- ReAct 让模型自主决定何时检索、何时直接回答
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import json
import copy
from loguru import logger

from config.settings import settings


class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class AgentStep:
    step_num: int
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[Any] = None
    timestamp: str = ""


@dataclass
class AgentState:
    """Agent 运行状态，支持快照和回滚"""
    session_id: str
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    context: list[dict] = field(default_factory=list)
    version: int = 0
    snapshots: list["AgentState"] = field(default_factory=list)

    def snapshot(self):
        """保存当前状态快照"""
        snap = copy.deepcopy(self)
        self.snapshots.append(snap)
        self.version += 1
        logger.debug(f"State snapshot saved: version={self.version}")

    def rollback(self) -> bool:
        """回滚到上一稳定态"""
        if not self.snapshots:
            logger.warning("No snapshot to rollback to")
            return False
        snap = self.snapshots.pop()
        self.steps = snap.steps
        self.context = snap.context
        self.version = snap.version
        logger.info(f"Rolled back to version {self.version}")
        return True

    def add_step(self, step: AgentStep):
        self.steps.append(step)

    def get_last_tool_call(self) -> Optional[AgentStep]:
        """获取上一次工具调用，用于重复检测"""
        for step in reversed(self.steps):
            if step.step_type == StepType.ACTION:
                return step
        return None


class ReActAgent:
    """ReAct 状态机 Agent"""

    def __init__(self, llm_client=None, retriever=None, tools: dict = None):
        self.llm = llm_client
        self.retriever = retriever
        self.tools = tools or {}
        self.max_steps = settings.agent_max_steps
        self.retry_limit = settings.agent_retry_limit

    async def run(self, query: str, session_id: str = "default") -> dict:
        """
        执行 ReAct 循环

        Returns:
            {"answer": str, "steps": list, "sources": list}
        """
        state = AgentState(session_id=session_id, query=query)
        state.snapshot()

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"Agent step {step_num}/{self.max_steps}")

            # 1. Thought: 决定下一步动作
            thought = await self._generate_thought(state)

            # 2. 检查是否可以直接回答
            if thought.get("should_answer", False):
                answer = await self._generate_final_answer(state)
                return {
                    "answer": answer,
                    "steps": [s.to_dict() if hasattr(s, 'to_dict') else s.__dict__
                              for s in state.steps],
                    "sources": state.context,
                }

            # 3. Action: 执行工具调用
            tool_name = thought.get("tool_name")
            tool_input = thought.get("tool_input", {})

            # 4. 重复工具调用检测
            last_call = state.get_last_tool_call()
            if last_call and last_call.tool_name == tool_name and \
               last_call.tool_input == tool_input:
                logger.warning(f"Detected repeated tool call: {tool_name}, stopping")
                answer = "抱歉，我无法找到相关信息，请尝试换个方式提问。"
                return {"answer": answer, "steps": [], "sources": []}

            # 5. 执行工具
            state.snapshot()
            observation = await self._execute_tool(
                tool_name, tool_input, state
            )

            # 6. 异常回滚
            if observation.get("error") and not observation.get("recovered"):
                state.rollback()
                if step_num >= self.retry_limit:
                    answer = observation.get("fallback",
                        "服务暂时不可用，请稍后重试。")
                    return {"answer": answer, "steps": [], "sources": []}

        # 超过最大步数
        answer = await self._generate_final_answer(state, force=True)
        return {
            "answer": answer,
            "steps": [],
            "sources": state.context,
        }

    async def _generate_thought(self, state: AgentState) -> dict:
        """生成下一步思考"""
        prompt = self._build_thought_prompt(state)
        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model_type="reasoning",
        )
        return self._parse_thought(response)

    async def _execute_tool(
        self, tool_name: str, tool_input: dict, state: AgentState
    ) -> dict:
        """执行工具调用，带三级容错"""
        tool = self.tools.get(tool_name)
        if not tool:
            return {"error": True, "message": f"Unknown tool: {tool_name}"}

        try:
            result = await tool.execute(tool_input)

            state.add_step(AgentStep(
                step_num=len(state.steps) + 1,
                step_type=StepType.ACTION,
                content=f"Called {tool_name}",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=result,
            ))

            if tool_name == "rag_search" and result.get("documents"):
                for doc in result["documents"]:
                    state.context.append({
                        "source": doc.chunk_id,
                        "text": doc.text,
                        "score": doc.score,
                    })

            return {"result": result, "error": False}

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")

            # 二级：本地纠错
            try:
                corrected_input = await self._local_correct(tool_input, str(e))
                result = await tool.execute(corrected_input)
                return {"result": result, "error": False, "recovered": True}
            except Exception:
                pass

            # 三级：兜底
            return {
                "error": True,
                "recovered": False,
                "fallback": "检索服务暂时不可用，请稍后重试。",
            }

    async def _local_correct(self, tool_input: dict, error: str) -> dict:
        """本地纠错：尝试修复工具输入"""
        corrected = tool_input.copy()
        if "query" in corrected and not corrected["query"].strip():
            corrected["query"] = "通用帮助"
        return corrected

    async def _generate_final_answer(
        self, state: AgentState, force: bool = False
    ) -> str:
        """生成最终答案"""
        context_text = "\n---\n".join(
            [c["text"] for c in state.context[:5]]
        )
        prompt = (
            f"用户问题：{state.query}\n\n"
            f"检索到的参考信息：\n{context_text}\n\n"
            "请基于参考信息回答用户问题。如果参考信息不足，"
            "请明确说明哪些部分无法回答。\n"
            "安全要求：不要在回答中包含手机号、身份证号等个人信息。"
        )
        return await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model_type="generation",
        )

    def _build_thought_prompt(self, state: AgentState) -> str:
        """构建 ReAct thought prompt"""
        history = ""
        for step in state.steps[-4:]:
            if step.step_type == StepType.ACTION:
                history += f"Action: {step.tool_name}({step.tool_input})\n"
            elif step.step_type == StepType.OBSERVATION:
                history += f"Observation: {step.tool_output}\n"

        return (
            f"用户问题：{state.query}\n\n"
            f"已有步骤：\n{history}\n\n"
            f"可用工具：{list(self.tools.keys())}\n"
            "请决定下一步：\n"
            '1. 如果已有足够信息，返回 {"should_answer": true}\n'
            '2. 如果需要检索，返回 {"should_answer": false, '
            '"tool_name": "rag_search", "tool_input": {"query": "..."}}\n'
        )

    def _parse_thought(self, response: str) -> dict:
        """解析 LLM 返回的 thought"""
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"should_answer": True}
