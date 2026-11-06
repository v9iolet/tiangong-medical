# src/agents/streaming.py
"""
筛选 Agent 流式输出中真正属于"最终回答"的 token。

LangGraph 的 astream(stream_mode="messages") 和 astream_events 会把同一次运行里所有 LLM 的 token 都推出来，
包括工具内部的查询改写、实体抽取、Cypher 生成，以及工具里嵌套调用的子 Agent。
fusion 等工具用 asyncio.gather 并发调用多个 LLM，这些 token 还会交错混在一起。
只有顶层 Agent 自己 model 节点产出的文本才是给用户看的回答。
"""

from typing import Any

from langchain_core.messages import AIMessageChunk


def is_agent_model_run(metadata: dict[str, Any]) -> bool:
    """
    判断事件是否来自顶层 Agent 的 model 节点。

    工具内部直接调用的 LLM 节点名是 tools；工具里嵌套的子 Agent 节点名虽然也是 model，
    但命名空间以 "tools:" 开头（形如 "tools:<id>|model:<id>"），按第一段区分。
    """
    namespace = metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns") or ""
    return metadata.get("langgraph_node") == "model" and namespace.split("|")[0].startswith("model:")


def is_answer_token(chunk: Any, metadata: dict[str, Any]) -> bool:
    """只放行顶层 model 节点的文本块；ToolMessage 和工具调用参数块都不是回答。"""
    return (
        isinstance(chunk, AIMessageChunk)
        and not chunk.tool_call_chunks
        and is_agent_model_run(metadata)
    )
