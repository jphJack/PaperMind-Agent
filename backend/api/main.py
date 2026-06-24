"""FastAPI 应用入口：挂载路由、配置 CORS、启动 uvicorn

启动方式：
    python run_backend.py
    或
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Paper Innovation Agent API",
    description="论文创新 Agent 异步接口：提交分析、SSE 进度推送、报告查询",
    version="1.0.0",
)

# 配置 CORS：允许前端 localhost:5173 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
app.include_router(router)


@app.on_event("startup")
async def on_startup() -> None:
    """应用启动日志"""
    logger.info("Paper Innovation Agent API 已启动")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
