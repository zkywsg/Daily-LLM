"""
配置管理模块
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基础配置
    APP_NAME: str = "Enterprise RAG+Agent System"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS配置
    CORS_ORIGINS: List[str] = ["*"]
    
    # API认证
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "your-jwt-secret")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # 模型配置
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    LLM_MODEL: str = "gpt-4"
    LLM_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LLM_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
    
    # 向量数据库配置
    VECTOR_STORE_TYPE: str = "milvus"  # milvus, chroma, faiss
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = "enterprise_docs"
    
    # Elasticsearch配置
    ES_HOST: str = os.getenv("ES_HOST", "localhost")
    ES_PORT: int = int(os.getenv("ES_PORT", "9200"))
    ES_INDEX: str = "documents"
    
    # Redis配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # RAG配置
    TOP_K_RETRIEVE: int = 20  # 初始检索数量
    TOP_K_RERANK: int = 5     # 重排序后数量
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    SCORE_THRESHOLD: float = 0.7
    
    # Agent配置
    MAX_TOOL_CALLS: int = 10
    AGENT_TIMEOUT: int = 120  # 秒
    
    # 记忆配置
    MAX_CONTEXT_LENGTH: int = 4096
    SESSION_EXPIRE_HOURS: int = 24
    
    # 安全配置
    MAX_QUERY_LENGTH: int = 2000
    RATE_LIMIT_PER_MINUTE: int = 60
    ENABLE_CONTENT_FILTER: bool = True
    
    # 监控配置
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()
