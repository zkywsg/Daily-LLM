"""
企业级RAG+Agent系统 - FastAPI主入口
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging

from src.api.routes import chat, agent, admin, health
from src.core.config import settings
from src.core.logging import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动事件
    logger.info("Starting Enterprise RAG+Agent System...")
    
    # 初始化连接池
    # await init_connections()
    
    yield
    
    # 关闭事件
    logger.info("Shutting down...")
    # await close_connections()


# 创建FastAPI应用
app = FastAPI(
    title="Enterprise RAG+Agent System",
    description="企业级知识助手系统，整合RAG检索增强和Agent工具调用能力",
    version="1.0.0",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(agent.router, prefix="/api/v1/agent", tags=["Agent"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Enterprise RAG+Agent System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/info")
async def info():
    """系统信息"""
    return {
        "app_name": settings.APP_NAME,
        "debug": settings.DEBUG,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "vector_store": settings.VECTOR_STORE_TYPE,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4
    )
