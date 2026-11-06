from neo4j import AsyncDriver, AsyncGraphDatabase

from src.core.config import get_settings

settings = get_settings()
_neo4j_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    """
    FastAPI Depends 注入用。
    返回可复用的 Neo4j 异步 Driver 单例。
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _neo4j_driver


async def check_neo4j_health() -> bool:
    """
    检查 Neo4j 连通性。
    """
    driver = get_neo4j_driver()
    await driver.verify_connectivity()
    return True


async def close_neo4j_driver() -> None:
    """关闭 Neo4j Driver。"""
    global _neo4j_driver
    if _neo4j_driver is not None:
        await _neo4j_driver.close()
        _neo4j_driver = None
