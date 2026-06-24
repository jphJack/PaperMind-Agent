"""Plan-and-Execute Controller Agent：动态规划执行路径并调度专职 Agent

核心职责：
- Plan 阶段：调用 LLM 生成动态执行计划（步骤列表），失败时回退默认七步
- Execute 阶段：逐步执行计划，支持并行调度、动态调整、异常降级
- Controller 自主决定调用哪个 Agent/工具，体现 Plan-and-Execute 框架
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Optional

from backend.agents.specialized import build_all_specialized_agents
from backend.config import settings
from backend.llm.client import DeepSeekClient
from backend.llm.json_utils import parse_json_safe
from backend.memory.manager import MemoryManager
from backend.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# Agent 名称 → 前端进度 stage 映射（与 StepStage 枚举值一致）
AGENT_STAGE_MAP: dict[str, str] = {
    "PaperParser": "step1_parse",
    "Extractor": "step2_extract",
    "Indexer": "step3_index",
    "GapAnalyzer": "step4_gap",
    "InnovationGenerator": "step5_innovation",
    "ExperimentDesigner": "step6_experiment",
    "Integration": "step7_integrate",
}

# 默认七步计划模板（LLM 生成失败时的兜底）
DEFAULT_PLAN: list[dict] = [
    {
        "step": 1,
        "agent": "PaperParser",
        "action": "扫描文件夹并解析所有 PDF 论文为结构化文本",
        "depends_on": [],
    },
    {
        "step": 2,
        "agent": "Extractor",
        "action": "对每篇已解析论文抽取结构化字段",
        "depends_on": [1],
    },
    {
        "step": 3,
        "agent": "Indexer",
        "action": "对已解析论文构建向量索引",
        "depends_on": [1],
    },
    {
        "step": 4,
        "agent": "GapAnalyzer",
        "action": "跨论文综合分析识别研究空白",
        "depends_on": [2],
    },
    {
        "step": 5,
        "agent": "InnovationGenerator",
        "action": "针对 Gap 生成创新点并筛选",
        "depends_on": [4],
    },
    {
        "step": 6,
        "agent": "ExperimentDesigner",
        "action": "为创新点设计实验方案",
        "depends_on": [5],
    },
    {
        "step": 7,
        "agent": "Integration",
        "action": "整合所有结果输出最终研究提案",
        "depends_on": [6],
    },
]

# progress_callback 类型别名
ProgressCallback = Callable[[str, str, str, float, Optional[dict]], Any]


class Controller:
    """Plan-and-Execute Controller：规划执行路径并逐步调度专职 Agent

    Controller 自主决定调用哪个 Agent/工具，而非硬编码固定流程。
    每步执行后根据结果动态调整剩余计划，支持并行调度与异常降级。
    """

    def __init__(
        self,
        client: DeepSeekClient,
        registry: ToolRegistry,
        memory: MemoryManager,
    ) -> None:
        self.client = client
        self.registry = registry
        self.memory = memory
        # 初始化 6 个专职 Agent
        self.agents = build_all_specialized_agents(client, registry)

        # 执行状态
        self.plan: list[dict] = []
        self.step_results: dict[int, dict] = {}
        self.tool_call_records: list[dict] = []
        self.degraded_steps: list[dict] = []

        # 步骤1 产出的已解析论文列表，供步骤2 并行抽取使用
        self._parsed_papers: list[dict] = []

    # ------------------------------------------------------------------ #
    # Plan 阶段
    # ------------------------------------------------------------------ #

    async def plan_task(self, folder_path: str) -> list[dict]:
        """Plan 阶段：调用 LLM 生成动态执行计划

        让 LLM 根据任务自主规划步骤顺序/增减，体现动态编排。
        LLM 调用失败或输出无效时回退到默认七步计划。

        Args:
            folder_path: 待分析的 PDF 文件夹路径

        Returns:
            计划列表，每步含 {step, agent, action, depends_on}
        """
        agent_descriptions = "\n".join(
            f"- {name}：{cls.__doc__ or ''}"
            for name, cls in [
                ("PaperParser", "扫描文件夹并解析 PDF"),
                ("Extractor", "单篇论文结构化抽取"),
                ("Indexer", "向量索引构建"),
                ("GapAnalyzer", "跨论文 Gap 识别"),
                ("InnovationGenerator", "创新点生成与筛选"),
                ("ExperimentDesigner", "实验方案设计"),
                ("Integration", "整合输出研究提案"),
            ]
        )

        prompt = (
            f"你是论文创新 Agent 的 Controller。请为以下任务生成动态执行计划。\n\n"
            f"任务：分析文件夹 {folder_path} 中的 PDF 论文，"
            f"识别研究空白，生成 2-3 个创新点并设计实验方案。\n\n"
            f"可用 Agent：\n{agent_descriptions}\n\n"
            f"请生成执行计划，输出 JSON 数组，每个元素含：\n"
            f"- step: 步骤序号（整数）\n"
            f"- agent: Agent 名称（上述之一）\n"
            f"- action: 该步骤具体任务描述\n"
            f"- depends_on: 依赖的步骤序号列表\n\n"
            f"你可以根据任务自主调整顺序或增减步骤。"
            f"默认链路：解析→抽取→索引→Gap分析→创新生成→实验设计→整合。\n"
            f"严格只输出 JSON 数组，不要其他文本。"
        )

        messages = [
            {
                "role": "system",
                "content": "你是 Agent 编排器，负责生成可执行的动态计划。只输出 JSON。",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat(messages=messages)
            content = response.choices[0].message.content or ""
            raw_plan = parse_json_safe(content)
            if isinstance(raw_plan, list) and len(raw_plan) > 0:
                normalized = self._normalize_plan(raw_plan)
                if normalized:
                    logger.info("LLM 生成计划成功，共 %d 步", len(normalized))
                    self.plan = normalized
                    return normalized
            logger.warning("LLM 计划输出无效，回退默认计划")
        except Exception as exc:
            logger.warning("LLM 生成计划失败，回退默认计划: %s", exc)

        # 兜底：默认七步计划
        default = [dict(s) for s in DEFAULT_PLAN]
        default[0]["action"] = f"扫描文件夹 {folder_path} 并解析所有 PDF 论文为结构化文本"
        self.plan = default
        return default

    def _normalize_plan(self, plan: list) -> list[dict]:
        """规范化 LLM 输出的计划：补全字段、过滤无效项"""
        normalized: list[dict] = []
        for idx, item in enumerate(plan):
            if not isinstance(item, dict):
                continue
            agent = str(item.get("agent", "")).strip()
            action = str(item.get("action", "")).strip()
            if not agent or not action:
                continue
            step = item.get("step", idx + 1)
            depends_on = item.get("depends_on", [])
            if not isinstance(depends_on, list):
                depends_on = []
            normalized.append(
                {
                    "step": step,
                    "agent": agent,
                    "action": action,
                    "depends_on": depends_on,
                }
            )
        return normalized

    # ------------------------------------------------------------------ #
    # Execute 阶段
    # ------------------------------------------------------------------ #

    async def execute_plan(
        self,
        plan: list[dict],
        progress_callback: Optional[ProgressCallback] = None,
        post_step_hook: Optional[Callable] = None,
    ) -> dict:
        """Execute 阶段：逐步执行计划，动态调整，并行调度，异常降级

        Args:
            plan: 计划列表
            progress_callback: 进度回调 async (stage, status, message, progress, payload)
            post_step_hook: 步骤后钩子 async (step, result) -> result，用于插入 Reflexion

        Returns:
            {steps_results, tool_call_records, degraded_steps}
        """
        remaining = list(plan)
        total = max(len(remaining), 1)
        completed = 0

        while remaining:
            step = remaining.pop(0)
            step_num = step.get("step", completed + 1)
            agent_name = step.get("agent", "")
            action = step.get("action", "")
            progress = completed / total

            # 推送 running 进度
            await self._emit_progress(
                progress_callback, agent_name, "running", action, progress
            )

            # 构建上下文
            context = self._build_context(step)

            try:
                result = await self._execute_step(step, context, progress_callback)

                # 步骤后钩子（供 WorkflowOrchestrator 插入 Reflexion 自校验）
                if post_step_hook is not None:
                    try:
                        hooked = await post_step_hook(step, result)
                        if hooked is not None:
                            result = hooked
                    except Exception as exc:
                        logger.warning("post_step_hook 执行失败: %s", exc)

                self.step_results[step_num] = result
                self.tool_call_records.extend(result.get("tool_calls", []))

                # 动态调整剩余计划
                remaining = self._adjust_plan(remaining, step, result)

                await self._emit_progress(
                    progress_callback,
                    agent_name,
                    "done",
                    f"步骤 {step_num}（{agent_name}）完成",
                    (completed + 1) / total,
                    {"step": step_num, "agent": agent_name},
                )
            except Exception as exc:
                logger.error(
                    "步骤 %d（%s）执行失败: %s", step_num, agent_name, exc, exc_info=True
                )
                self.step_results[step_num] = {
                    "agent": agent_name,
                    "result": "",
                    "tool_calls": [],
                    "error": str(exc),
                }

                # 评估是否降级
                degraded = self._evaluate_degradation(agent_name, str(exc))
                if degraded:
                    self.degraded_steps.append(degraded)
                    await self._emit_progress(
                        progress_callback,
                        agent_name,
                        "degraded",
                        f"步骤 {step_num} 降级: {degraded['reason']}",
                        (completed + 1) / total,
                        {"step": step_num, "degradation": degraded},
                    )
                else:
                    await self._emit_progress(
                        progress_callback,
                        agent_name,
                        "failed",
                        f"步骤 {step_num} 失败: {exc}",
                        (completed + 1) / total,
                        {"step": step_num, "error": str(exc)},
                    )

            completed += 1
            total = completed + len(remaining)

        return {
            "steps_results": self.step_results,
            "tool_call_records": self.tool_call_records,
            "degraded_steps": self.degraded_steps,
        }

    async def _execute_step(
        self,
        step: dict,
        context: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        """执行单个步骤：根据 agent 名称选择执行策略"""
        agent_name = step.get("agent", "")
        action = step.get("action", "")

        # Integration 步骤：Controller 直接调用 LLM 整合
        if agent_name == "Integration" or agent_name not in self.agents:
            return await self._execute_integration(action, context)

        # Extractor 步骤：若已有多篇解析论文，并行抽取
        if agent_name == "Extractor" and self._parsed_papers:
            return await self._execute_extraction_parallel(action, context)

        # 其他步骤：单 Agent 执行
        return await self._execute_single(agent_name, action, context)

    async def _execute_single(
        self, agent_name: str, action: str, context: str
    ) -> dict:
        """单 Agent 执行步骤"""
        agent = self.agents.get(agent_name)
        if agent is None:
            raise ValueError(f"未知 Agent: {agent_name}")

        result = await agent.run(task=action, context=context)

        # PaperParser 完成后，收集已解析论文供后续并行抽取
        if agent_name == "PaperParser":
            await self._collect_parsed_papers(result, action)

        return result

    async def _execute_extraction_parallel(
        self, action: str, context: str
    ) -> dict:
        """并行抽取多篇论文结构化字段

        使用 asyncio.gather + semaphore(MAX_CONCURRENT_PAPERS) 并行调度。
        每篇论文独立处理，失败隔离不影响其余。
        """
        if not self._parsed_papers:
            # 无已解析论文，降级为单 Agent 执行
            logger.warning("无已解析论文，Extractor 降级为单 Agent 执行")
            return await self._execute_single("Extractor", action, context)

        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_PAPERS)
        agent = self.agents["Extractor"]

        async def extract_one(paper: dict) -> dict:
            """单篇论文抽取任务：直接调用 extract_paper_structure 工具（绕过模型 Function Calling）

            说明：参数已由 Controller 确定（paper_path + parsed_paper），
            让模型再选一次工具反而会出现参数传递错误（DeepSeek 有时把整段 JSON 当字符串传），
            浪费迭代次数。这里直接用 ToolLoop 调用工具，再让模型基于结果生成摘要。
            """
            async with semaphore:
                paper_path = paper.get("path", "")
                try:
                    # 1) 直接调用 extract_paper_structure 工具（参数已知）
                    extract_fn = self.registry.get_callable("extract_paper_structure")
                    if extract_fn is None:
                        raise RuntimeError("extract_paper_structure 工具未注册")

                    import time as _time
                    _t0 = _time.perf_counter()
                    extraction_result = await self._invoke_tool(
                        "extract_paper_structure",
                        paper_path=paper_path,
                        parsed_paper=paper,
                    )
                    _duration = round(_time.perf_counter() - _t0, 3)

                    # 2) 记录工具调用
                    self.tool_call_records.append(
                        {
                            "tool_name": "extract_paper_structure",
                            "args_summary": json.dumps(
                                {"paper_path": paper_path}, ensure_ascii=False
                            )[:200],
                            "success": True,
                            "duration_sec": _duration,
                            "error": None,
                        }
                    )

                    # 3) 让 LLM 基于抽取结果生成一段摘要（用于前端展示 + 下游 Agent）
                    if isinstance(extraction_result, str):
                        extraction_text = extraction_result
                    else:
                        extraction_text = json.dumps(
                            extraction_result, ensure_ascii=False, default=str
                        )

                    summary_messages = [
                        {
                            "role": "system",
                            "content": (
                                "你是论文结构化抽取结果的整理助手。"
                                "请基于给定的抽取结果，输出 200 字以内的核心摘要，"
                                "包含 title/task_problem/method/key_contributions 四项要点。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"论文路径：{paper_path}\n\n抽取结果：\n{extraction_text[:6000]}",
                        },
                    ]
                    try:
                        resp = await self.client.chat(messages=summary_messages)
                        summary_text = resp.choices[0].message.content or extraction_text[:2000]
                    except Exception:
                        # LLM 摘要失败时直接用原始抽取结果
                        summary_text = extraction_text[:2000]

                    return {
                        "agent": "Extractor",
                        "result": summary_text,
                        "tool_calls": [
                            {
                                "tool_name": "extract_paper_structure",
                                "args_summary": json.dumps(
                                    {"paper_path": paper_path}, ensure_ascii=False
                                )[:200],
                                "success": True,
                                "duration_sec": _duration,
                                "error": None,
                                "result": extraction_result,  # 供 ReportAgent 聚合使用
                            }
                        ],
                        "paper_path": paper_path,
                        "extraction": extraction_result,
                    }
                except Exception as exc:
                    logger.warning("论文 %s 抽取失败: %s", paper_path, exc)
                    return {
                        "agent": "Extractor",
                        "result": "",
                        "tool_calls": [],
                        "error": str(exc),
                        "paper_path": paper_path,
                    }

        # 并行调度所有论文抽取
        results = await asyncio.gather(
            *[extract_one(p) for p in self._parsed_papers]
        )

        # 合并结果与工具调用记录
        all_tool_calls: list[dict] = []
        result_parts: list[str] = []
        success_count = 0
        for r in results:
            all_tool_calls.extend(r.get("tool_calls", []))
            result_text = r.get("result", "")
            if result_text:
                result_parts.append(result_text)
                success_count += 1

        logger.info(
            "并行抽取完成: %d/%d 篇成功", success_count, len(self._parsed_papers)
        )

        return {
            "agent": "Extractor",
            "result": "\n---\n".join(result_parts),
            "tool_calls": all_tool_calls,
            "papers_count": len(results),
            "success_count": success_count,
            "individual_results": results,
        }

    async def _execute_integration(self, action: str, context: str) -> dict:
        """Integration 步骤：Controller 直接调用 LLM 整合所有步骤结果"""
        parts: list[str] = []
        for step_num, result in self.step_results.items():
            result_text = (
                result.get("result", "") if isinstance(result, dict) else str(result)
            )
            if result_text:
                parts.append(f"## 步骤 {step_num} 结果\n{result_text[:3000]}")

        all_results = "\n\n".join(parts) if parts else context

        messages = [
            {
                "role": "system",
                "content": (
                    "你是研究提案整合专家。请整合各步骤结果，"
                    "输出结构化研究提案（背景综述/创新点论证/实验方案/参考文献溯源）。"
                ),
            },
            {
                "role": "user",
                "content": f"任务：{action}\n\n各步骤结果：\n{all_results[:10000]}",
            },
        ]

        try:
            response = await self.client.chat(messages=messages)
            result_text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("整合步骤失败: %s", exc)
            result_text = ""

        return {
            "agent": "Integration",
            "result": result_text,
            "tool_calls": [],
        }

    async def _collect_parsed_papers(self, result: dict, action: str) -> None:
        """从 PaperParser 结果中收集已解析论文，供后续并行抽取使用

        优先从结果文本解析 JSON；失败则直接调用 scan_folder + parse_pdf 工具获取。
        """
        # 尝试从结果文本中解析 JSON 论文列表
        text = result.get("result", "")
        papers = parse_json_safe(text)
        if isinstance(papers, list) and papers:
            self._parsed_papers = papers
            logger.info("从 PaperParser 结果收集到 %d 篇论文", len(papers))
            return

        # 兜底：从 action 提取文件夹路径，直接调用工具获取结构化论文
        folder_path = self._extract_folder_path(action)
        if not folder_path:
            logger.warning("无法从 action 提取文件夹路径，跳过论文收集")
            return

        try:
            scan_result = await self._invoke_tool("scan_folder", folder_path=folder_path)
            pdf_files = scan_result.get("pdf_files", [])
            logger.info("扫描到 %d 个 PDF 文件，开始解析", len(pdf_files))

            # 并行解析所有 PDF
            semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_PAPERS)

            async def parse_one(pdf_path: str) -> dict:
                async with semaphore:
                    try:
                        return await self._invoke_tool("parse_pdf", file_path=pdf_path)
                    except Exception as exc:
                        logger.warning("解析 %s 失败: %s", pdf_path, exc)
                        return {"path": pdf_path, "error": str(exc)}

            parsed = await asyncio.gather(*[parse_one(p) for p in pdf_files])
            # 过滤掉解析失败的
            self._parsed_papers = [p for p in parsed if "error" not in p]
            logger.info("成功解析 %d 篇论文", len(self._parsed_papers))
        except Exception as exc:
            logger.error("收集已解析论文失败: %s", exc)
            self._parsed_papers = []

    def _extract_folder_path(self, action: str) -> str:
        """从 action 描述中提取文件夹路径"""
        # 匹配 "扫描文件夹 {path}" 或 "文件夹 {path}" 模式
        match = re.search(r"文件夹\s+([^\s，,。]+)", action)
        if match:
            return match.group(1)
        return ""

    async def _invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """调用注册表中的工具（自动区分同步/异步）"""
        fn = self.registry.get_callable(tool_name)
        if fn is None:
            raise ValueError(f"工具 {tool_name} 未注册")
        if asyncio.iscoroutinefunction(fn):
            return await fn(**kwargs)
        return await asyncio.to_thread(fn, **kwargs)

    # ------------------------------------------------------------------ #
    # 动态调整与降级
    # ------------------------------------------------------------------ #

    def _adjust_plan(
        self, remaining: list[dict], step: dict, result: dict
    ) -> list[dict]:
        """根据当前步骤结果动态调整剩余计划

        - 抽取完全失败 → 跳过依赖它的 Gap 分析及后续步骤
        - 索引失败 → 不跳过后续步骤，但标记降级（GapAnalyzer 仅用结构化字段）
        """
        agent_name = step.get("agent", "")
        has_error = isinstance(result, dict) and bool(result.get("error"))

        if agent_name == "Extractor":
            # 抽取结果为空或全部失败 → 跳过后续依赖步骤
            success_count = result.get("success_count", 0) if isinstance(result, dict) else 0
            papers_count = result.get("papers_count", 0) if isinstance(result, dict) else 0
            if has_error or (papers_count > 0 and success_count == 0):
                logger.warning("抽取无有效结果，跳过 Gap 分析及后续步骤")
                remaining = [
                    s
                    for s in remaining
                    if s.get("agent")
                    not in ("GapAnalyzer", "InnovationGenerator", "ExperimentDesigner")
                ]

        if agent_name == "Indexer" and has_error:
            # 索引失败，后续 GapAnalyzer 仅用结构化字段（不跳过，仅记录降级）
            logger.info("索引失败，后续步骤将仅用结构化字段（跳过 RAG）")

        return remaining

    def _evaluate_degradation(
        self, agent_name: str, error: str
    ) -> Optional[dict]:
        """评估步骤失败是否可降级继续

        Returns:
            降级记录 dict，或 None（不可降级）
        """
        if agent_name == "Indexer":
            return {
                "step": agent_name,
                "reason": f"向量索引构建失败: {error}；降级为仅用结构化字段，跳过 RAG",
                "strategy": "skip_rag",
            }
        if agent_name == "InnovationGenerator":
            err_lower = error.lower()
            if "web_search" in err_lower or "search" in err_lower or "联网" in error:
                return {
                    "step": agent_name,
                    "reason": f"联网搜索失败: {error}；降级为跳过新颖性去重",
                    "strategy": "skip_novelty_check",
                }
        return None

    # ------------------------------------------------------------------ #
    # 辅助方法
    # ------------------------------------------------------------------ #

    def _build_context(self, step: dict) -> str:
        """构建步骤上下文：从依赖步骤的结果中提取"""
        depends_on = step.get("depends_on", [])
        if not depends_on:
            return ""
        parts: list[str] = []
        for dep in depends_on:
            if dep in self.step_results:
                result = self.step_results[dep]
                result_str = json.dumps(result, ensure_ascii=False, default=str)[
                    :5000
                ]
                parts.append(f"步骤 {dep} 结果：\n{result_str}")
        return "\n\n".join(parts)

    async def _emit_progress(
        self,
        callback: Optional[ProgressCallback],
        agent_name: str,
        status: str,
        message: str,
        progress: float,
        payload: Optional[dict] = None,
    ) -> None:
        """推送进度事件到回调"""
        if callback is None:
            return
        stage = AGENT_STAGE_MAP.get(agent_name, agent_name)
        try:
            await callback(stage, status, message, progress, payload)
        except Exception as exc:
            logger.warning("进度回调执行失败: %s", exc)

    # ------------------------------------------------------------------ #
    # 完整流程
    # ------------------------------------------------------------------ #

    async def run(
        self,
        folder_path: str,
        progress_callback: Optional[ProgressCallback] = None,
        post_step_hook: Optional[Callable] = None,
    ) -> dict:
        """完整流程：plan_task → execute_plan → 返回最终结果

        Args:
            folder_path: PDF 文件夹路径
            progress_callback: 进度回调
            post_step_hook: 步骤后钩子（供 Reflexion 插入）

        Returns:
            {steps_results, tool_call_records, degraded_steps, plan}
        """
        # 重置状态
        self.step_results = {}
        self.tool_call_records = []
        self.degraded_steps = []
        self._parsed_papers = []

        # Plan 阶段
        self.plan = await self.plan_task(folder_path)

        # Execute 阶段
        result = await self.execute_plan(
            self.plan, progress_callback, post_step_hook
        )

        # 附加计划信息，供后续整合输出 Agent 使用
        result["plan"] = self.plan
        return result
