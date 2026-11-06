from src.core.config import get_settings
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

settings = get_settings()

# 1. 创建引擎(连接池)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG, # 调试模式下，会打印出所有的 SQL 语句
    pool_size=10, # 连接池大小
    max_overflow=20, # 超过连接池大小后，最多可以创建的连接数
    pool_timeout=30, # 连接池获取连接的超时时间，单位秒
    pool_recycle=60 * 5, # 连接池连接的最大空闲时间，单位秒
    pool_pre_ping=True, # 连接池在获取连接时，会先测试连接是否可用

)


# 2. 创建会话工厂
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False, # 提交事务后，会话不会过期
)

# 3. 定义异步获取数据库会话连接
async def get_db() -> AsyncSession:
    """ FastAPI 依赖注入，获取数据库会话连接 Depends(get_db) """
    async with AsyncSessionLocal() as session:
        try:
            yield session  # 返回的session别人用
            await session.commit() # 用完后自动提交事务
        except Exception as e:
            await session.rollback() # 出错后回滚事务
            raise e # 抛出异常，让FastAPI处理