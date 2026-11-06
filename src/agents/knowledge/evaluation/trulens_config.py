"""
TruLens 会话配置 + LLM Provider + Dashboard 启动
"""
from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM
from src.core.config import get_settings
from src.agents.knowledge.evaluation.provider import CompatibleLiteLLM

settings = get_settings()


def get_trulens_session() -> TruSession:
    """评估结果存入独立的 PostgreSQL 数据库"""
    return TruSession(
        database_url=f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
    )


def get_llm_provider() -> LiteLLM:
    """所有评估指标通过 OpenAI 兼容协议使用 DEEPSEEK_* 配置。"""
    settings = get_settings()
    if not settings.DEEPSEEK_MODEL or not settings.DEEPSEEK_API_KEY or not settings.DEEPSEEK_BASE_URL:
        raise ValueError("评估模型需要配置 DEEPSEEK_MODEL、DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL")
    return CompatibleLiteLLM(
        model_engine=f"openai/{settings.DEEPSEEK_MODEL}",
        completion_kwargs={
            "api_key": settings.DEEPSEEK_API_KEY,
            "api_base": settings.DEEPSEEK_BASE_URL,
        },
    )


def launch_dashboard(port: int = 8501):
    """启动 TruLens Streamlit Dashboard"""
    from trulens.dashboard import run_dashboard
    session = get_trulens_session()
    run_dashboard(session=session, port=port)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    launch_dashboard(port)
