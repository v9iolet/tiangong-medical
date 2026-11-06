"""单轮执行问诊图，多轮状态由调用方传回。

首轮加载上下文并检查急症；后续根据待回答症状选择解析回答或重新提取。
澄清/追问后结束本次执行，收到下一条用户消息再继续。
"""

from __future__ import annotations
import json
from typing import Any

from langchain_community.embeddings import DashScopeEmbeddings
from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pymilvus import MilvusClient

from src.infra.milvus_store import MilvusStore
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.neo4j_client import get_neo4j_driver
from src.core.config import get_settings, get_worker_llm, get_embedding_model
from src.agents.context import make_thread_id
from src.agents.inquiry.state import (
    InquiryState, InquiryPhase, InquiryHandoffPayload, PatientContext
)
from src.agents.inquiry.symptom_normalizer import (
    normalize_symptoms, humanize_symptoms
)
from src.agents.inquiry.neo4j_queries import (
    query_candidate_diseases, enrich_candidate_details, get_pending_symptoms
)
from src.agents.inquiry.confidence import apply_context_weights, check_convergence
from src.agents.inquiry.prompts import (
    CLARIFY_PROMPT, ASK_SYMPTOMS_PROMPT, PARSE_ANSWER_PROMPT,
    CONCLUSION_PROMPT, EMERGENCY_CHECK_PROMPT
)

settings = get_settings()

# ── 依赖注入容器（组装图时绑定，节点不自行创建客户端） ──────────────────────
class InquiryDeps:
    def __init__(self, llm, neo4j_driver, embedding_model, milvus_client, db_session, store):
        self.llm = llm
        self.neo4j_driver = neo4j_driver
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.db_session = db_session
        self.store = store  # MilvusStore 实例，用于写回长期记忆


# ════════════════════════════════════════════════════════════════════════
# 节点函数（每个节点接收 state，返回 state 的部分更新）
# ════════════════════════════════════════════════════════════════════════
# 返回字典由图合入 state：未返回的字段保留原值，普通字段覆盖，messages 按 ID 合并。
# 症状列表需先与历史合并再返回；返回 {} 表示不更新状态，不表示结束执行。

async def node_load_context(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点①：加载患者上下文。
    从 PostgreSQL 加载档案和近期问诊信息，保留调用方传入的长期记忆。
    仅在第一轮（round==0）执行，后续轮次跳过。
    """
    if state.round > 0:
        logger.debug("节点①加载患者上下文 非首轮，跳过上下文加载 (round={})", state.round)
        return {}  # 保留上下文，固定边仍指向 check_emergency

    logger.info("节点①加载患者上下文 开始加载患者上下文 | patient_id={} session_id={}",
                state.patient_context.patient_id, state.session_id)

    from src.agents.inquiry.db_queries import load_patient_context
    # patient_id 对应 patients.id，与 user_id 分开传递。
    patient_id = state.patient_context.patient_id
    # 近期问诊中的诊断会合入既往病史，供候选疾病加权。
    patient_ctx = await load_patient_context(
        patient_id=patient_id,
        db=deps.db_session,
    )

    # TODO：如需节点内检索长期记忆，在此接入 Milvus；当前沿用传入值。


    merged_ctx = PatientContext(
        patient_id=state.patient_context.patient_id,  # 保留本次问诊关联的患者 ID
        age=patient_ctx.age,
        gender=patient_ctx.gender,
        allergy_history=patient_ctx.allergy_history,
        medical_history=patient_ctx.medical_history,
        long_term_memories=state.patient_context.long_term_memories,
    )
    logger.info("节点①加载患者上下文 上下文加载完成 | age={} gender={} medical_history={} allergy={}",
                merged_ctx.age, merged_ctx.gender,
                len(merged_ctx.medical_history), len(merged_ctx.allergy_history))
    # 仅替换患者上下文，phase/round 不变；固定边进入 check_emergency。
    return {"patient_context": merged_ctx}


async def node_check_emergency(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点②：急症识别。
    仅在第一轮执行；识别到急症则直接回复就医提示并结束，不走候选查询和保存节点。
    """
    if state.round > 0:
        logger.debug("节点②急症识别 非首轮，跳过急症检查")
        return {}  # 不更新状态，仍由 route_after_emergency 决定去向

    logger.info("节点②急症识别 开始急症识别检查")
    last_user_msg = ""
    for msg in reversed(state.messages):  # 只检查最新用户输入，跳过 AI 消息
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {}  # 保留原阶段，仍执行急症节点后的路由

    prompt = EMERGENCY_CHECK_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        if result.get("is_emergency"):
            logger.warning("节点②急症识别 ⚠️ 检测到急症！原因: {}", result.get('reason', ''))
            emergency_reply = (
                "⚠️ 根据您描述的症状，这可能是紧急情况！\n\n"
                f"原因：{result.get('reason', '存在急症风险')}\n\n"
                "**请立即前往最近医院的急诊科就诊，或拨打 120 急救电话。**\n\n"
                "不要等待，请立即行动！"
            )
            # 追加急症提示并置 END；路由直接结束本次执行，不再生成普通结论。
            return {
                "phase": InquiryPhase.END,
                "messages": [AIMessage(content=emergency_reply)],
            }
        else:
            logger.info("节点②急症识别 未检测到急症，继续正常问诊流程")
    except Exception as e:
        # 解析失败时不修改阶段，当前流程仍会继续提取症状。
        logger.warning(f"急症识别解析失败: {e}")
    return {}  # 正常首轮保留初始 CLARIFY 阶段，路由继续进入 extract_symptoms


async def node_extract_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点③：症状标准化。
    从最新的用户消息中提取症状，经三层标准化后合并到 confirmed_symptoms。
    """
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        logger.warning("节点③症状标准化 未找到用户消息，跳过症状提取")
        return {}  # 不重设 phase，后续路由沿用原阶段

    logger.info("节点③症状标准化 开始提取症状 | 用户输入: {!r}", last_user_msg[:80])

    # LLM 提取术语 → Neo4j 精确匹配 → Milvus 语义兜底。
    result = await normalize_symptoms(
        user_input=last_user_msg,
        llm=deps.llm,
        neo4j_driver=deps.neo4j_driver,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
    )

    # 累计合并并去重；未匹配项单独保留，不参与候选疾病查询。
    new_confirmed = list(set(state.confirmed_symptoms) | set(result["all_standard"]))
    # 当前仅累加未匹配项，后续匹配成功也不会自动清除历史项。
    new_unmatched = list(set(state.unmatched_symptoms) | set(result["unmatched"]))

    logger.info("节点③症状标准化 提取完成 | 标准化症状={} 未匹配={} 累计确认={}",
                result["all_standard"], result["unmatched"], new_confirmed)

    # 有任一累计确认症状即可查图谱，不要求本轮全部匹配；一个都没有才澄清。
    if new_confirmed:
        # 覆盖两类累计症状并置 GRAPH_QUERY；更新后的 state 交给 query_neo4j。
        return {
            "confirmed_symptoms": new_confirmed,
            "unmatched_symptoms": new_unmatched,
            "phase": InquiryPhase.GRAPH_QUERY,  # route_after_extract 据此选择查询节点
        }
    else:
        logger.info("节点③症状标准化 无法提取明确症状，进入澄清流程")
        # 保存未匹配项并置 CLARIFY；确认列表仍为空，随后进入 clarify。
        return {
            "unmatched_symptoms": new_unmatched,
            "phase": InquiryPhase.CLARIFY,
        }


async def node_clarify(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点④：澄清模糊描述。
    累计没有可用于查图谱的标准症状时，引导补充描述；不是逐项处理未匹配症状。
    """
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    logger.info("节点④澄清模糊描述 用户描述模糊，发起澄清引导 | round={}", state.round)
    prompt = CLARIFY_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    logger.debug("节点④澄清模糊描述 澄清回复已生成，等待用户下一轮输入")
    # round 只在发出澄清或症状追问时递增，不是节点执行次数。
    # 追加澄清消息，保留 CLARIFY；本次执行结束，下一轮重新提取症状。
    return {
        "round": state.round + 1,
        "messages": [AIMessage(content=response.content)],
    }


async def node_query_neo4j(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑤：查询 Neo4j 候选疾病。
    用已确认症状查候选疾病，补充详情，应用上下文权重。
    """
    logger.info("节点⑤查询候选疾病 查询候选疾病 | 确认症状={}", state.confirmed_symptoms)
    candidates = await query_candidate_diseases(
        confirmed_symptoms=state.confirmed_symptoms,
        neo4j_driver=deps.neo4j_driver,
    )
    if candidates:
        # 每次查询都补齐详情，再结合病史、长期记忆和否认症状重新评分排序。
        candidates = await enrich_candidate_details(candidates, deps.neo4j_driver)
        candidates = apply_context_weights(
            candidates,
            state.patient_context,
            state.denied_symptoms
        )
        logger.info("节点⑤查询候选疾病 候选疾病 top5: {}",
                    [(c.name, round(c.confidence, 3)) for c in candidates[:5]])
    else:
        logger.warning("节点⑤查询候选疾病 未找到候选疾病，将强制结束问诊")

    # 达到轮次上限、首位置信度达标或明显领先时收敛；无候选另行兜底。
    should_conclude, force_conclude = check_convergence(candidates, state.round)
    logger.info("节点⑤查询候选疾病 收敛判断 | should_conclude={} force_conclude={} round={}",
                should_conclude, force_conclude, state.round)

    if should_conclude:
        # 替换本轮候选及强制结束标记，置 CONCLUDE 后进入 conclude。
        return {
            "candidate_diseases": candidates,
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": force_conclude,
        }
    elif not candidates:
        # 清空旧候选并标记强制结束，置 CONCLUDE 后输出无结果提示。
        return {
            "candidate_diseases": [],
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": True,
        }
    else:
        # 有候选疾病，且未收敛，继续追询症状。
        # 替换候选并置 SYMPTOM_CONFIRM，进入 ask_symptoms；症状和轮次不变。
        return {
            "candidate_diseases": candidates,
            "phase": InquiryPhase.SYMPTOM_CONFIRM,
        }


async def node_ask_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑥：生成追问话术。
    从候选疾病中选出尚未问过、也未确认或否认的 ≤3 个症状进行追问。
    """
    logger.info("节点⑥生成追问话术 生成追问 | round={} 已问症状={}", state.round, state.asked_symptoms)
    pending = await get_pending_symptoms(
        candidates=state.candidate_diseases,
        confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms,
        asked_symptoms=state.asked_symptoms,
    )

    # 优先询问在候选症状列表中出现次数较少的症状，以帮助区分候选。
    top_symptoms = [s for s, _ in pending[:3]]

    if not top_symptoms:
        # 没有更多可问的症状，直接收敛
        logger.info("节点⑥生成追问话术 无更多可追问症状，直接进入结论")
        # 只置 CONCLUDE，沿用当前候选生成结论，不增加轮次或追加追问消息。
        return {"phase": InquiryPhase.CONCLUDE}

    logger.info("节点⑥生成追问话术 本轮追问症状: {}", top_symptoms)
    # 展示给用户时口语化，状态仍保存标准词，便于下一轮解析和查询。
    human_symptoms = await humanize_symptoms(top_symptoms, deps.llm)

    prompt = ASK_SYMPTOMS_PROMPT.format(
        symptoms_to_ask="\n".join(f"- {s}" for s in human_symptoms)
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])

    # 更新轮次、已问/待答列表并追加追问；保留 SYMPTOM_CONFIRM，本次执行结束。
    return {
        "round": state.round + 1,
        "asked_symptoms": state.asked_symptoms + top_symptoms,  # 累计已问，避免重复追问
        "pending_ask_symptoms": top_symptoms,  # 本轮待回答；下一次调用从 parse_answer 进入
        "messages": [AIMessage(content=response.content)],
    }


async def node_parse_answer(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑦：解析用户对追问的回答。
    判断用户确认/否认了哪些症状，更新 confirmed_symptoms 和 denied_symptoms。
    """
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg or not state.pending_ask_symptoms:
        logger.debug("节点⑦解析用户回答 无用户消息或无待解析症状，跳过")
        return {}  # 保留全部字段，route_after_parse 仍进入 query_neo4j

    logger.info("节点⑦解析用户回答 解析用户回答 | 待解析症状={} 用户回答={!r}",
                state.pending_ask_symptoms, last_user_msg[:80])

    # 将上一轮追问的标准症状交给 LLM，解释“有/没有”等依赖上下文的回答。
    prompt = PARSE_ANSWER_PROMPT.format(
        asked_symptoms=", ".join(state.pending_ask_symptoms),
        user_answer=last_user_msg,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)
        # 按模型结果分别累加；当前不校验是否属于追问项，也不消解确认/否认冲突。
        new_confirmed = list(
            set(state.confirmed_symptoms) | set(parsed.get("confirmed", []))
        )
        new_denied = list(
            set(state.denied_symptoms) | set(parsed.get("denied", []))
        )
        logger.info("节点⑦解析用户回答 解析结果 | 新增确认={} 新增否认={}",
                    parsed.get("confirmed", []), parsed.get("denied", []))
    except Exception as e:
        # 解析失败时保留已有判断，后续仍按原症状重新查询候选。
        logger.warning(f"解析用户回答失败: {e}")
        new_confirmed = state.confirmed_symptoms
        new_denied = state.denied_symptoms

    # 更新确认/否认列表并清空待答项；phase/round 不变，路由直接重新查询候选。
    return {
        "confirmed_symptoms": new_confirmed,
        "denied_symptoms": new_denied,
        "pending_ask_symptoms": [],  # 无论解析是否成功都清空；asked_symptoms 仍保留
    }


async def node_conclude(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑧：生成诊断结论。
    根据候选排序生成结论文案和移交数据包；无候选时只输出就医建议。
    """
    candidates = state.candidate_diseases
    if not candidates:
        logger.warning("节点⑧生成诊断结论 无候选疾病，返回无结果提示")
        no_result_msg = (
            "根据您描述的症状，我暂时无法在知识库中找到匹配的疾病。\n\n"
            "这可能是因为症状较为罕见，或者需要更多信息。\n"
            "建议您直接前往医院就诊，由医生进行面诊。"
        )
        # 追加无结果提示并置 END；不生成移交数据包，也不进入保存节点。
        return {
            "phase": InquiryPhase.END,
            "messages": [AIMessage(content=no_result_msg)],
        }

    # 候选已按上下文加权后的置信度排序，首位作为主要候选，其余最多保留四个。
    top1 = candidates[0]
    suspected = [c.name for c in candidates[1:5]]
    logger.info("节点⑧生成诊断结论 生成诊断结论 | 主诊断={} 置信度={:.3f} 科室={} 疑似={}",
                top1.name, top1.confidence, top1.department, suspected)

    prompt = CONCLUSION_PROMPT.format(
        confirmed_symptoms="、".join(state.confirmed_symptoms) or "暂无",
        primary_disease=top1.name,
        confidence=top1.confidence,
        suspected_diseases="、".join(suspected) if suspected else "无",
        department=top1.department or "综合内科",
        checks="、".join(top1.checks[:5]) if top1.checks else "暂无",
        force_conclude=state.force_conclude,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])

    # 保留否认和未匹配症状供后续接手；本节点只构建数据包，不执行挂号。
    handoff = InquiryHandoffPayload(
        user_id=state.user_id,
        thread_id=state.thread_id,
        patient_id=state.patient_context.patient_id,
        patient_context=state.patient_context.model_dump(),
        confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms,
        unmatched_symptoms=state.unmatched_symptoms,
        primary_disease=top1.name,
        primary_confidence=top1.confidence,
        suspected_diseases=suspected,
        department=top1.department or "综合内科",
        recommended_checks=top1.checks[:5],
        total_rounds=state.round,
        session_id=state.session_id,
    )

    # 写入移交数据包并追加结论，置 HANDOFF；图内下一步是保存记录，不是挂号。
    return {
        "phase": InquiryPhase.HANDOFF,
        "handoff_payload": handoff,
        "messages": [AIMessage(content=response.content)],
    }


async def node_save_record(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑨：保存问诊记录到 PostgreSQL。
    仅在结论节点生成移交数据包后执行，不等待挂号确认。
    """
    if not state.handoff_payload:
        logger.warning("节点⑨保存问诊记录 无 handoff_payload，跳过保存")
        # 仅置 END，保留已有消息和症状，固定边结束本次执行。
        return {"phase": InquiryPhase.END}

    from src.agents.inquiry.db_queries import save_consultation_record
    payload = state.handoff_payload
    # 主诉摘要取当前列表前五项，不代表用户原文或症状重要性排序。
    chief_complaint = "、".join(state.confirmed_symptoms[:5])
    # 只使用显式患者 ID，不能把用户身份当作患者外键。
    patient_id = payload.patient_id
    if patient_id is None:
        logger.warning("节点⑨保存问诊记录 未关联 patient_id，跳过数据库保存 | thread={}",
                       state.thread_id)
        return {"phase": InquiryPhase.END}

    logger.info("节点⑨保存问诊记录 保存问诊记录 | patient_id={} diagnosis={} department={}",
                patient_id, payload.primary_disease, payload.department)

    import time as _time

    if state.handoff_payload and state.patient_context.patient_id:
        # 把本次诊断结论存入长期记忆
        # 下次问诊时 search_memory 能检索到，触发 +0.10 置信度加权
        memory_content = (
            f"曾被诊断为「{state.handoff_payload.primary_disease}」，"
            f"确认症状：{'、'.join(state.handoff_payload.confirmed_symptoms[:5])}"
        )
        # deps.store 需要在 InquiryDeps 中传入 MilvusStore 实例
        await deps.store.aput(
            namespace=("users", str(state.patient_context.patient_id), "memories"),
            key=f"diagnosis_{state.session_id}",
            value={"content": memory_content, "timestamp": _time.time()},
        )

        logger.info("节点⑨保存问诊记录 保存长期记忆 | memory={}", memory_content)



    # 保存函数失败可返回 None；此处不检查返回值，仍结束问诊图。
    await save_consultation_record(
        patient_id=patient_id,
        session_id=payload.session_id,
        chief_complaint=chief_complaint,
        diagnosis=payload.primary_disease,
        department_name=payload.department,
        urgency_level="normal",
        db=deps.db_session,
    )
    logger.info("节点⑨保存问诊记录 问诊记录保存完成，流程结束")
    # HANDOFF → END；保留结论消息和移交数据包，供调用方读取，不追加回复。
    return {"phase": InquiryPhase.END}





# ════════════════════════════════════════════════════════════════════════
# 路由函数（决定下一个节点）
# ════════════════════════════════════════════════════════════════════════
# 路由读取节点更新后的 state，返回节点名或 END，不修改 state。
# phase 是业务阶段而非当前节点名；实际去向由固定边或条件路由决定。

def route_dispatcher(state: InquiryState) -> str:
    """入口分发：首轮走完整流程，后续轮次根据上一轮状态决定入口。"""
    if state.round == 0:
        return "load_context"
    # 上一轮是追问症状（pending_ask_symptoms 非空）→ 解析用户回答
    if state.pending_ask_symptoms:
        return "parse_answer"
    # 没有待回答症状时，按新描述提取（通常是用户回复澄清问题）。
    return "extract_symptoms"


def route_after_emergency(state: InquiryState) -> str:
    """急症检查后的路由：急症直接结束，否则提取症状。"""
    if state.phase == InquiryPhase.END:
        return END
    return "extract_symptoms"


def route_after_extract(state: InquiryState) -> str:
    """沿用提取节点的阶段判断，不单独检查未匹配列表。"""
    if state.phase == InquiryPhase.GRAPH_QUERY:
        return "query_neo4j"
    return "clarify"


def route_after_neo4j(state: InquiryState) -> str:
    """Neo4j 查询后的路由：收敛则结论，否则追问。"""
    if state.phase == InquiryPhase.CONCLUDE:
        return "conclude"
    return "ask_symptoms"


def route_after_ask(state: InquiryState) -> str:
    """追问后的路由：没有更多症状可问则直接结论。"""
    if state.phase == InquiryPhase.CONCLUDE:
        return "conclude"
    return END  # 等待用户回答（下一轮消息进来后从 parse_answer 继续）


def route_after_parse(state: InquiryState) -> str:
    """解析回答后：重新查 Neo4j 更新候选疾病。"""
    return "query_neo4j"


def route_after_conclude(state: InquiryState) -> str:
    """有移交数据包则进入保存；无候选的结论已置 END，直接结束。"""
    if state.phase == InquiryPhase.END:
        return END
    return "save_record"


# ════════════════════════════════════════════════════════════════════════
# 图的组装
# ════════════════════════════════════════════════════════════════════════

def build_inquiry_graph(deps: InquiryDeps):
    """
    构建并编译问诊 StateGraph。
    deps 通过闭包注入到每个节点函数中。
    """
    # 用闭包把 deps 绑定到节点函数
    async def _load_context(state):    return await node_load_context(state, deps)
    async def _check_emergency(state): return await node_check_emergency(state, deps)
    async def _extract_symptoms(state):return await node_extract_symptoms(state, deps)
    async def _clarify(state):         return await node_clarify(state, deps)
    async def _query_neo4j(state):     return await node_query_neo4j(state, deps)
    async def _ask_symptoms(state):    return await node_ask_symptoms(state, deps)
    async def _parse_answer(state):    return await node_parse_answer(state, deps)
    async def _conclude(state):        return await node_conclude(state, deps)
    async def _save_record(state):     return await node_save_record(state, deps)

    graph = StateGraph(InquiryState)

    # 注册节点（dispatcher 是无操作的分发节点，仅用于路由）
    graph.add_node("dispatcher",       lambda state: {})
    graph.add_node("load_context",     _load_context)
    graph.add_node("check_emergency",  _check_emergency)
    graph.add_node("extract_symptoms", _extract_symptoms)
    graph.add_node("clarify",          _clarify)
    graph.add_node("query_neo4j",      _query_neo4j)
    graph.add_node("ask_symptoms",     _ask_symptoms)
    graph.add_node("parse_answer",     _parse_answer)
    graph.add_node("conclude",         _conclude)
    graph.add_node("save_record",      _save_record)

    # 每次调用都从 dispatcher 进入，以轮次和待回答症状判断接续位置。
    graph.set_entry_point("dispatcher")

    # END 只终止本次图执行；澄清/追问时保留业务阶段，下一条消息再接续。
    graph.add_edge("load_context", "check_emergency")
    graph.add_edge("clarify", END)          # 澄清后等待用户回复
    graph.add_edge("save_record", END)

    # 注册条件边：先合入节点返回值，再调用路由函数，将结果映射为下一个节点。
    graph.add_conditional_edges("dispatcher",       route_dispatcher,
                                 {"load_context": "load_context", "parse_answer": "parse_answer",
                                  "extract_symptoms": "extract_symptoms"})
    graph.add_conditional_edges("check_emergency",  route_after_emergency,
                                 {"extract_symptoms": "extract_symptoms", END: END})
    graph.add_conditional_edges("extract_symptoms", route_after_extract,
                                 {"query_neo4j": "query_neo4j", "clarify": "clarify"})
    graph.add_conditional_edges("query_neo4j",      route_after_neo4j,
                                 {"conclude": "conclude", "ask_symptoms": "ask_symptoms"})
    graph.add_conditional_edges("ask_symptoms",     route_after_ask,
                                 {"conclude": "conclude", END: END})
    graph.add_conditional_edges("parse_answer",     route_after_parse,
                                 {"query_neo4j": "query_neo4j"})
    graph.add_conditional_edges("conclude",         route_after_conclude,
                                 {"save_record": "save_record", END: END})
    # 未配置 checkpointer；多轮状态依赖调用方保存并传回 existing_state。
    return graph.compile()


# ════════════════════════════════════════════════════════════════════════
# 对外接口：供 call_inquiry_agent 工具调用
# ════════════════════════════════════════════════════════════════════════

async def run_inquiry(
    user_message: str,
    user_id: str,
    session_id: str,
    deps: InquiryDeps,
    existing_state: InquiryState | None = None,
    patient_id: int | None = None,
    long_term_memories: list[str] | None = None,
) -> tuple[str, InquiryState]:
    """
    执行一轮问诊对话；首轮和后续轮次统一传入用户、会话身份。

    user_id/session_id：本次请求身份；thread_id 在此统一生成，不由调用方单独传入。
    patient_id：独立的 patients.id，仅用于首轮加载档案和保存记录。
    existing_state：上一轮状态；身份必须与请求一致，旧状态缺失的身份字段会补齐。
    后续省略 patient_id 时沿用状态中的患者；显式传入不同患者则拒绝，需另开会话。
    long_term_memories：首轮传入的长期记忆摘要，后续沿用已有状态。

    Returns:
        (assistant_reply, new_state)
    """
    thread_id = make_thread_id(user_id, session_id)
    identity = {"user_id": user_id, "session_id": session_id, "thread_id": thread_id}

    if existing_state is None:
        state = InquiryState(
            **identity,
            patient_context=PatientContext(
                patient_id=patient_id,
                long_term_memories=long_term_memories or [],
            ),
        )
    else:
        # 兼容 Redis 中未保存全部身份字段的旧状态，但不允许跨用户/会话复用。
        legacy_session_is_thread = (
            not existing_state.user_id
            and existing_state.session_id == thread_id
            and existing_state.thread_id in ("", thread_id)
        )
        for field, expected in identity.items():
            stored = getattr(existing_state, field)
            # 旧版本曾把复合 thread_id 写入 session_id；仅对没有 user_id 的旧状态兼容。
            if field == "session_id" and legacy_session_is_thread:
                continue
            if stored and stored != expected:
                raise ValueError(f"问诊状态的 {field} 与当前请求不一致")
        if patient_id is not None and patient_id != existing_state.patient_context.patient_id:
            raise ValueError("同一问诊会话不能更换 patient_id，请新建 session_id")
        # 不修改调用方传入的上一轮状态，身份字段在返回状态中统一补齐。
        state = existing_state.model_copy(deep=True, update=identity)

    state.messages.append(HumanMessage(content=user_message))

    logger.info("▶ run_inquiry 开始 | thread={} session={} user={} round={} entry={}",
                state.thread_id, state.session_id, state.user_id, state.round,
                route_dispatcher(state))

    graph = build_inquiry_graph(deps)
    config = {"configurable": {"thread_id": state.thread_id}}
    result = await graph.ainvoke(state, config=config)

    # result 是本次执行合并后的完整状态，不只是最后一个节点返回的字典。
    new_state = InquiryState(**result) if isinstance(result, dict) else result

    # 取最后一条 AI 消息作为回复
    reply = ""
    for msg in reversed(new_state.messages):
        if isinstance(msg, AIMessage):
            reply = msg.content
            break

    logger.info("◀ run_inquiry 完成 | thread={} phase={} round={} reply_len={}",
                thread_id, new_state.phase, new_state.round, len(reply))
    # 回复供展示，完整状态供调用方保存；等待用户回复时，下次作为 existing_state 传回。
    return reply, new_state



def build_inquiry_deps(db_session=None) -> InquiryDeps:
    """
    构建问诊依赖注入容器的工厂函数。
    供 call_inquiry_agent 工具和 FastAPI 路由共用，避免重复代码。
    """
    llm = get_worker_llm(temperature=0.3)
    embedding_model = get_embedding_model()
    neo4j_driver = get_neo4j_driver()
    get_milvus_client_alias()  # 确保连接已建立
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    # 新增：传入 MilvusStore 实例，用于写回长期记忆

    store = MilvusStore(
        alias=get_milvus_client_alias(),
        embeddings=embedding_model,
        dims=1024,
    )

    return InquiryDeps(
        llm=llm,
        neo4j_driver=neo4j_driver,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        db_session=db_session,
        store=store,  # 新增
    )
