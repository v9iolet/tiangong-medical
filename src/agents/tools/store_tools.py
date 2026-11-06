from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.agents.context import UserContext


def get_user_id(runtime: ToolRuntime[UserContext]):
    # 用户身份直接来自同一个调用上下文，不再从 thread_id 拆分推断。
    return runtime.context.user_id if runtime.context is not None else None


@tool
async def save_memory(
    content: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """
    将重要信息保存到长期记忆中。
    当用户提到重要的个人信息、偏好、病史、过敏史、药物史、手术史等需要跨会话记住的内容时调用。

    Args:
        content: 要记住的内容，用一句话描述
    """

    print(f"🔧执行：保存长期记忆：{content} ")
    user_id = get_user_id(runtime)
    if not user_id:
        return "无法获取用户ID。"

    import time
    key = f"memory_{int(time.time())}"
    await runtime.store.aput(
        namespace=("users", user_id, "memories"),
        key=key,
        value={"content": content, "timestamp": time.time()},
    )
    return f"已记住：{content}"


@tool
async def search_memory(
    query: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """
    从长期记忆中检索与问题相关的历史信息。
    当需要回忆用户之前说过的内容、历史病情、偏好等时调用。

    Args:
        query: 检索关键词或问题
    """

    user_id = get_user_id(runtime)
    print(f"🔧执行：搜索长期记忆：{query}; 用户id：{user_id}")
    if not user_id:
        return "无法获取用户ID。"

    results = await runtime.store.asearch(
        ("users", user_id, "memories"),
        query=query,
        limit=5,
    )
    if not results:
        return "没有找到相关记忆。"

    memories = [f"- {item.value['content']}" for item in results]
    return "相关历史记忆：\n" + "\n".join(memories)
