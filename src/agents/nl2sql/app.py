"""
ChatBI 可视化面板 - Gradio 界面
支持：自然语言查询 → SQL → 图表 + 多轮对话 + 角色权限
"""
from __future__ import annotations
import asyncio
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.core.config import get_settings
from src.infra.database import AsyncSessionLocal
from src.nl2sql.engine import run_query, ConversationContext, QueryResult
from src.nl2sql.chart_advisor import recommend_chart, render_chart

load_dotenv()
settings = get_settings()

# 每个 session 维护独立的对话上下文
_contexts: dict[str, ConversationContext] = {}


def _get_llm():
    return ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.1,
    )


def _get_context(session_id: str) -> ConversationContext:
    if session_id not in _contexts:
        _contexts[session_id] = ConversationContext()
    return _contexts[session_id]


async def _process_query(
    question: str,
    role: str,
    department_id: int | None,
    session_id: str,
) -> tuple[str, pd.DataFrame | None, object | None, str]:
    """
    处理一次查询，返回 (summary, dataframe, plotly_figure, sql)
    """
    if not question.strip():
        return "请输入问题", None, None, ""

    llm = _get_llm()
    context = _get_context(session_id)

    async with AsyncSessionLocal() as db:
        result = await run_query(
            question=question,
            llm=llm,
            db=db,
            role=role,
            department_id=department_id,
            context=context,
        )

    if not result.success:
        return f"查询失败: {result.error}", None, None, result.sql

    df = pd.DataFrame(result.data) if result.data else None

    chart_config = await recommend_chart(question, result.data, result.columns, llm)
    fig = render_chart(result.data, chart_config)

    description = chart_config.get("description", "")
    summary = result.summary
    if description:
        summary = f"{summary}\n\n📊 {description}"

    return summary, df, fig, result.sql


def query_handler(question, role, department_id, session_id, chat_history):
    """Gradio 回调：处理用户输入"""
    dept_id = int(department_id) if department_id and department_id.strip() else None

    summary, df, fig, sql = asyncio.run(
        _process_query(question, role, dept_id, session_id)
    )

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": question})

    response = summary
    if sql:
        response += f"\n\n```sql\n{sql}\n```"
    chat_history.append({"role": "assistant", "content": response})

    return chat_history, df, fig, sql


def clear_handler(session_id):
    """清空对话上下文"""
    if session_id in _contexts:
        del _contexts[session_id]
    return [], None, None, ""


# ── Gradio UI ────────────────────────────────────────────────────────────

with gr.Blocks(title="天宫医疗 - ChatBI") as demo:
    gr.Markdown("# 天宫医疗 - ChatBI 数据分析面板")
    gr.Markdown("输入自然语言问题，自动生成 SQL 查询并可视化结果。支持多轮追问。")

    session_id = gr.State("default_session")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 设置")
            role = gr.Dropdown(
                choices=["admin", "operator", "doctor"],
                value="operator",
                label="角色",
            )
            department_id = gr.Textbox(
                label="科室 ID（doctor 角色需填写）",
                placeholder="如：1",
                value="",
            )
            gr.Markdown("---")
            gr.Markdown("### 示例问题")
            gr.Markdown("""
- 各科室问诊量排名
- 库存不足 50 的药品有哪些
- 本月每天的问诊量趋势
- OTC 药品和处方药的数量对比
- 各科室常见疾病 TOP5
- 药品价格分布
            """)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=400)
            with gr.Row():
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例：各科室问诊量排名前10",
                    scale=4,
                )
                submit_btn = gr.Button("查询", variant="primary", scale=1)
                clear_btn = gr.Button("清空", scale=1)

            with gr.Tabs():
                with gr.Tab("图表"):
                    chart_output = gr.Plot(label="可视化图表")
                with gr.Tab("数据表"):
                    table_output = gr.Dataframe(label="查询结果")
                with gr.Tab("SQL"):
                    sql_output = gr.Code(label="生成的 SQL", language="sql")

    submit_btn.click(
        query_handler,
        inputs=[question_input, role, department_id, session_id, chatbot],
        outputs=[chatbot, table_output, chart_output, sql_output],
    )
    question_input.submit(
        query_handler,
        inputs=[question_input, role, department_id, session_id, chatbot],
        outputs=[chatbot, table_output, chart_output, sql_output],
    )
    clear_btn.click(
        clear_handler,
        inputs=[session_id],
        outputs=[chatbot, table_output, chart_output, sql_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
