"""FastAPI 路由：分析任务提交、SSE 进度推送、报告查询、健康检查

后台分析用 asyncio.create_task 启动，不阻塞 API 响应；
SSE 用 sse_starlette.EventSourceResponse，从 task_manager 事件历史推送。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.agents.report_agent import ReportAgent
from backend.api.tasks import task_manager
from backend.config import settings
from backend.evaluator.evaluator import Evaluator
from backend.llm.client import DeepSeekClient
from backend.memory.manager import MemoryManager
from backend.orchestrator.workflow import WorkflowOrchestrator
from backend.tools.registry import build_default_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["paper-innovation"])


# ---------------------------------------------------------------------- #
# 请求模型
# ---------------------------------------------------------------------- #

class AnalyzeRequest(BaseModel):
    """POST /api/analyze 请求体"""
    folder_path: str = Field(description="待分析的 PDF 文件夹路径")


# ---------------------------------------------------------------------- #
# 后台分析函数
# ---------------------------------------------------------------------- #

async def run_analysis(task_id: str, folder_path: str) -> None:
    """后台分析主流程：编排工作流 → 生成报告 → 生成评估 → 更新任务状态

    Args:
        task_id: 任务 ID
        folder_path: PDF 文件夹路径
    """
    logger.info("启动后台分析: task_id=%s, folder=%s", task_id, folder_path)
    try:
        # 标记为运行中
        task_manager.update_task(
            task_id, status="running", progress=0.0, current_stage=None
        )

        # 初始化组件：DeepSeek 客户端 + 工具注册表 + 记忆管理器
        client = DeepSeekClient()
        registry = build_default_registry()
        memory = MemoryManager()
        orchestrator = WorkflowOrchestrator(client, registry, memory)

        # 进度回调：更新任务状态 + 追加事件
        async def progress_callback(
            stage: str,
            status: str,
            message: str,
            progress: float,
            payload: Optional[dict] = None,
        ) -> None:
            task_manager.update_task(
                task_id, progress=progress, current_stage=stage
            )
            task_manager.add_event(
                task_id,
                {
                    "stage": stage,
                    "status": status,
                    "message": message,
                    "progress": progress,
                    "payload": payload,
                },
            )

        # 执行工作流流水线
        pipeline_output = await orchestrator.run_pipeline(
            folder_path, progress_callback
        )
        final_results = pipeline_output.get("final_results", {}) or {}
        evaluation_data = pipeline_output.get("evaluation_data", {}) or {}

        # 用 ReportAgent 生成最终报告
        report_agent = ReportAgent(client=client)
        report = await report_agent.generate_report(final_results)

        # 用 Evaluator 生成评估
        evaluator = Evaluator(client=client)
        evaluation = await evaluator.evaluate(
            final_results,
            evaluation_data.get("tool_call_records", []),
            evaluation_data.get("confidence_stats", {}),
        )

        # 存储最终结果，标记完成
        task_manager.update_task(
            task_id,
            status="completed",
            progress=1.0,
            result={
                "markdown": report.get("markdown", ""),
                "report": report,
                "evaluation": evaluation,
            },
        )
        logger.info("任务完成: task_id=%s", task_id)

    except Exception as exc:
        logger.error(
            "任务失败: task_id=%s, error=%s", task_id, exc, exc_info=True
        )
        task_manager.update_task(task_id, status="failed", error=str(exc))


# ---------------------------------------------------------------------- #
# 路由
# ---------------------------------------------------------------------- #

@router.post("/analyze")
async def analyze(request: AnalyzeRequest) -> dict:
    """提交分析任务：创建 task_id 并后台启动，立即返回 task_id

    Args:
        request: 含 folder_path 的请求体

    Returns:
        {"task_id": str}
    """
    if not request.folder_path:
        raise HTTPException(status_code=400, detail="folder_path 不能为空")

    # 创建任务
    task_id = task_manager.create_task(request.folder_path)

    # 后台启动分析（不阻塞响应）
    asyncio.create_task(run_analysis(task_id, request.folder_path))

    return {"task_id": task_id}


@router.get("/progress/{task_id}")
async def progress(task_id: str) -> EventSourceResponse:
    """SSE 流：推送任务进度事件，任务结束时推送 done 事件并关闭

    Args:
        task_id: 任务 ID

    Returns:
        EventSourceResponse：逐条推送 {stage, status, message, progress, payload}
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    async def event_generator():
        """SSE 事件生成器：从事件历史增量推送，任务结束发 done"""
        last_index = 0
        while True:
            # 1. 排空所有未发送事件
            events = task_manager.get_events(task_id)
            while last_index < len(events):
                event = events[last_index]
                last_index += 1
                yield {
                    "event": "progress",
                    "data": json.dumps(event, ensure_ascii=False, default=str),
                }

            # 2. 检查任务是否结束
            current = task_manager.get_task(task_id)
            if current is None:
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"message": "任务不存在"}, ensure_ascii=False
                    ),
                }
                return
            if current["status"] in ("completed", "failed"):
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "status": current["status"],
                            "task_id": task_id,
                            "progress": current.get("progress", 0.0),
                            "error": current.get("error"),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                }
                return

            # 3. 清除通知标志后再次检查事件，避免竞态漏推
            notifier = task_manager.get_notifier(task_id)
            if notifier is not None:
                notifier.clear()
                # 清除后若有新事件，回到循环顶部发送
                events = task_manager.get_events(task_id)
                if last_index < len(events):
                    continue
                # 等待新事件到达（带超时作为 keepalive）
                try:
                    await asyncio.wait_for(notifier.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    # 超时发送 keepalive 注释，保持连接
                    yield {"comment": "keepalive"}
            else:
                # 无通知器则短睡后重试
                await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator(), ping=15)


@router.get("/report/{task_id}")
async def report(task_id: str) -> dict:
    """获取最终报告

    Args:
        task_id: 任务 ID

    Returns:
        {markdown, report, evaluation, status}
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    status = task.get("status")
    result = task.get("result") or {}

    if status == "completed":
        return {
            "markdown": result.get("markdown", ""),
            "report": result.get("report", {}),
            "evaluation": result.get("evaluation", {}),
            "status": status,
        }
    elif status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"任务失败: {task.get('error', '未知错误')}",
        )
    else:
        # pending / running：返回当前进度状态
        return {
            "markdown": "",
            "report": {},
            "evaluation": {},
            "status": status,
            "progress": task.get("progress", 0.0),
            "current_stage": task.get("current_stage"),
        }


@router.get("/tasks/{task_id}")
async def task_status(task_id: str) -> dict:
    """获取任务状态

    Args:
        task_id: 任务 ID

    Returns:
        {task_id, status, progress, current_stage}
    """
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    return {
        "task_id": task_id,
        "status": task.get("status", "pending"),
        "progress": task.get("progress", 0.0),
        "current_stage": task.get("current_stage"),
    }


@router.get("/health")
async def health() -> dict:
    """健康检查"""
    return {"status": "ok", "model": settings.DEEPSEEK_MODEL}
