"""读取项目 .env，并按主/子 Agent 职责创建模型。"""
import os
from functools import lru_cache
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "tiangong-agent"
    APP_ENV: str = "dev"
    APP_DEBUG: bool = True

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "medical"
    DB_PASSWORD: str = "medical123"
    DB_NAME: str = "medical_db"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "knowledge-docs"
    MINIO_SECURE: bool = False

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "medical123"

    # 主 Agent（Supervisor + 会话摘要）：DashScope 的兼容接口。
    DASHSCOPE_API_KEY: str = ""
    BASE_URL_CHAT: str = ""
    CHAT_MODEL: str = "qwen3.8-flash"

    # 子 Agent / 问诊图：独立模型和密钥，未指定地址时沿用当前兼容网关。
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-flash"
    DEEPSEEK_BASE_URL: str = ""

    # 向量和视觉模型与主/子 Agent 的文本模型独立。
    EMBEDDING_MODEL: str = "qwen3.7-text-embedding"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_DIMENSIONS: int = Field(default=1024, gt=0)
    VL_MODEL: str = "qwen3.8-omni-flash"

    # MinerU 4.x 自部署 V1 API（mineru-kit api-server），地址不含 /v1，如 http://127.0.0.1:8000
    MINERU_API_URL: str = ""
    MINERU_API_KEY: str = ""  # 服务端以 --api-key 启动时填写，本地匿名访问留空
    MINERU_TIER: str = ""  # flash / basic / standard / advanced；留空使用服务端 --tier 的默认档位
    PARSED_ARTIFACT_DIR: str = "data/parsed"  # 相对路径基于项目根目录
    MINERU_TIMEOUT: int = 1000  # 单个文档解析的总等待时长，单位秒；本地 standard 档位较慢

    JWT_SECRET_KEY: str = ""  # Required for JWT issuance/verification; never use a source-code default.

    # ChatBI: defaults are fail-closed; copy only approved deployment grants.
    CHATBI_ENABLED: bool = False
    CHATBI_IDENTITY_PROVIDER: str = ""  # server-side module:function(subject) -> user
    CHATBI_ROLE_PERMISSIONS: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    # Optional server-side identity for local Gradio testing only. API auth is unchanged.
    CHATBI_LOCAL_TEST_PRINCIPAL: dict[str, object] = Field(default_factory=dict)
    CHATBI_MAX_ROWS: int = Field(default=500, ge=1, le=5000)
    CHATBI_POOL_SIZE: int = Field(default=5, ge=1, le=30)
    CHATBI_SESSION_TTL: int = Field(default=1800, ge=60)
    CHATBI_MAX_SESSIONS: int = Field(default=1000, ge=1)
    CHATBI_MODEL_TIMEOUT: int = Field(default=60, ge=1, le=180)

    # 日志配置
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # 指定环境变量文件
    model_config = {
        "env_file": Path(__file__).resolve().parents[2] / ".env",
        "env_file_encoding": "utf-8",
    }

# 保存到内存缓存中。以后直接获取。
@lru_cache  # lru 把对象实例保存到内存中。这是一种单例的实现
def get_settings() -> Settings:
    return Settings()


def get_supervisor_llm(temperature: float = 0.3) -> ChatOpenAI:
    """主 Agent 和摘要模型只读取 CHAT_MODEL 这组配置。"""
    settings = get_settings()
    if not settings.CHAT_MODEL or not settings.DASHSCOPE_API_KEY or not settings.BASE_URL_CHAT:
        raise ValueError("主 Agent 需要配置 CHAT_MODEL、DASHSCOPE_API_KEY 和 BASE_URL_CHAT")
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}}
    )


def get_worker_llm(temperature: float = 0.3) -> ChatDeepSeek:
    """所有子 Agent / 问诊节点只读取 DEEPSEEK_MODEL 这组配置。"""
    settings = get_settings()
    if not settings.DEEPSEEK_MODEL or not settings.DEEPSEEK_API_KEY or not settings.DEEPSEEK_BASE_URL:
        raise ValueError(
            "子 Agent 需要配置 DEEPSEEK_MODEL、DEEPSEEK_API_KEY，"
            "以及 DEEPSEEK_BASE_URL 或 BASE_URL_CHAT"
        )
    return ChatDeepSeek(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        # 当前安装的 ChatDeepSeek 客户端使用 api_base 构造请求地址。
        api_base=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}}
    )


def get_embedding_model() -> Embeddings:
    """向量接口独立配置；兼容硅基流动的完整 /embeddings 地址。"""
    settings = get_settings()
    if settings.EMBEDDING_API_KEY or settings.EMBEDDING_BASE_URL:
        if not settings.EMBEDDING_MODEL or not settings.EMBEDDING_API_KEY or not settings.EMBEDDING_BASE_URL:
            raise ValueError("向量模型需要配置 EMBEDDING_MODEL、EMBEDDING_API_KEY 和 EMBEDDING_BASE_URL")
        base_url = settings.EMBEDDING_BASE_URL.rstrip("/").removesuffix("/embeddings")
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY,
            base_url=base_url,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            # 兼容接口接收原始文本，不能用 OpenAI tokenizer 转成 token IDs。
            check_embedding_ctx_length=False,
            chunk_size=20,
            request_timeout=60,
            # 批量入库并发高时可能撞 TPM 限流，429 由 openai SDK 按 Retry-After 或指数退避重试
            max_retries=5,
        )
    return DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


def get_llm(temperature: float = 0.3) -> ChatDeepSeek:
    """兼容旧调用；新子 Agent 请显式使用 get_worker_llm。"""
    return get_worker_llm(temperature=temperature)
