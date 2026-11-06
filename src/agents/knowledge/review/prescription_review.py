# src/agents/knowledge/review/prescription_review.py
"""
处方审核：对用户描述的处方做用药安全校验，并生成审核报告。

流程：LLM 解析处方为结构化 JSON → 并行执行剂量 / 配伍 / 过敏 / 重复用药校验 → LLM 汇总生成审核报告。
剂量校验走文档 RAG（药品说明书），配伍与过敏校验走 GraphRAG（医学知识图谱）。
"""

from __future__ import annotations
import asyncio
import json
from dataclasses import replace
from loguru import logger
from src.agents.knowledge.retrieval.diagnostics import observe_source, report_issue, report_source_count
from src.agents.knowledge.retrieval.evidence import Evidence
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient

from src.agents.knowledge.prompts import (
    PRESCRIPTION_PARSE_PROMPT, PRESCRIPTION_REPORT_PROMPT,
)
from src.agents.knowledge.retrieval.doc_rag import search_docs_raw, format_doc_context
from src.agents.knowledge.retrieval.options import DocRetrievalOptions
from src.agents.knowledge.retrieval.graph_rag import search_graph_raw


async def _parse_prescription(question: str, llm: BaseChatModel) -> dict:
    """
    从自然语言中提取处方结构化信息。

    Returns:
        {"drugs": [{"name", "dosage", "frequency", "route"}, ...],
         "patient_info": {"allergies": [...], "diseases": [...], "age", "gender"}}
        结构由 PRESCRIPTION_PARSE_PROMPT 约定；解析失败时返回空药品列表，由调用方提示用户补充信息。
    """
    prompt = PRESCRIPTION_PARSE_PROMPT.format(question=question)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        # 兼容模型把 JSON 包在 ```json ... ``` 代码块里的情况
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("drugs"), list):
            raise ValueError("invalid_prescription")
        report_source_count(len(parsed["drugs"]))
        return parsed
    except Exception as e:
        logger.warning(f"处方解析失败: {e}")
        report_issue("prescription_parse_failed", fatal=True)
        report_source_count(0)
        return {"drugs": [], "patient_info": {"allergies": [], "diseases": []}}


def dosage_options(options: DocRetrievalOptions | None = None) -> DocRetrievalOptions:
    """剂量检索固定预算与 HyDE/hybrid 约束；保留集合、分块和重排策略。"""
    return replace(options or DocRetrievalOptions(), top_k=5, context_top_k=2,
                   use_hyde=False, use_hybrid=True)


async def _check_dosage(
    drug_name: str,
    dosage: str | None,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    doc_options: DocRetrievalOptions | None = None,
) -> dict:
    """
    剂量校验：查说明书推荐剂量范围。

    这里只负责检索说明书中的用法用量片段，不做数值比对；
    处方剂量是否超出范围由报告生成阶段的 LLM 对照 reference 判断。
    """
    query = f"{drug_name} 用法用量 推荐剂量"
    # 只检索药品说明书；用法用量信息集中，召回和精排数量都取较小值
    hits = await search_docs_raw(
        query, embedding_model, milvus_client,
        llm=None, options=dosage_options(doc_options),
        doc_type="drug_instruction",
    )
    report_source_count(len(hits))
    context = format_doc_context(hits) if hits else "未找到说明书"
    return {
        "type": "dosage",
        "drug": drug_name,
        "input_dosage": dosage,
        # 截断参考片段，避免多药处方时校验结果撑爆报告 prompt
        "reference": context[:500],
        "source": hits[0]["doc_name"] if hits else "无",
    }


async def _check_interaction(
    drug_names: list[str],
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> dict:
    """
    配伍校验：查药物间相互作用。

    把全部药品拼成一个自然语言问题交给 GraphRAG，由 LLM 生成 Cypher 查询药物之间的关系，
    而不是逐对两两查询；图谱中无记录不代表一定安全，只表示未发现已知禁忌。
    """
    if len(drug_names) < 2:
        report_source_count(0, skipped=True)
        return {"type": "interaction", "drugs": drug_names, "result": "单药无需配伍校验", "records": []}

    query = f"{'、'.join(drug_names)} 之间是否有药物相互作用或配伍禁忌"
    records = await search_graph_raw(query, neo4j_driver, llm)
    report_source_count(len(records))
    return {
        "type": "interaction",
        "drugs": drug_names,
        # 最多保留 10 条图谱记录，控制报告 prompt 长度
        "records": records[:10],
        "result": json.dumps(records[:10], ensure_ascii=False) if records else "知识图谱中未发现配伍禁忌记录",
    }


async def _check_allergy(
    drug_names: list[str],
    allergies: list[str],
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> dict:
    """
    过敏校验（成分级匹配）：
    1. 字符串匹配：药品名 vs 过敏史
    2. GraphRAG 成分级匹配：查药品所属类别，与过敏史交叉比对
    """
    conflicts = []
    # 统一转小写，兼容英文药名大小写不一致（如 Penicillin / penicillin）
    allergy_set = {a.lower() for a in allergies}

    # 1. 字符串匹配：先查完全一致，再查互相包含（如过敏史“青霉素”命中“青霉素V钾片”）
    for drug in drug_names:
        if drug.lower() in allergy_set:
            conflicts.append({"drug": drug, "allergy": drug, "level": "exact_match"})
            continue
        for allergy in allergies:
            if allergy.lower() in drug.lower() or drug.lower() in allergy.lower():
                conflicts.append({"drug": drug, "allergy": allergy, "level": "name_match"})
                break

    graph_count = 0
    # 2. GraphRAG 成分级匹配：覆盖药名不含过敏物质名称的情况（如对青霉素过敏但处方为阿莫西林）
    if allergies and drug_names:
        query = (
            f"查询以下药品的药物类别和成分信息：{'、'.join(drug_names)}。"
            f"患者对以下物质过敏：{'、'.join(allergies)}。"
            f"判断是否存在交叉过敏风险。"
        )
        records = await search_graph_raw(query, neo4j_driver, llm)
        graph_count = len(records)
        if records:
            for r in records[:10]:
                # 图谱记录的列名由 LLM 生成的 Cypher 决定，结构不固定，
                # 因此整条记录序列化成字符串后做包含匹配
                r_str = json.dumps(r, ensure_ascii=False).lower()
                for allergy in allergies:
                    # 同一条记录里同时出现过敏物质和处方药品，视为存在交叉过敏风险
                    if allergy.lower() in r_str:
                        drug_in_record = next(
                            (d for d in drug_names if d.lower() in r_str), None,
                        )
                        # 已在字符串匹配阶段命中的药品不重复记录
                        if drug_in_record and not any(
                            c["drug"] == drug_in_record for c in conflicts
                        ):
                            conflicts.append({
                                "drug": drug_in_record,
                                "allergy": allergy,
                                "level": "component_match",
                                # 保留图谱原始记录片段作为依据，供报告标注来源
                                "evidence": json.dumps(r, ensure_ascii=False)[:200],
                            })

    report_source_count(len(conflicts) + graph_count, skipped=not bool(allergies and drug_names))
    return {
        "type": "allergy",
        "allergies": allergies,
        "conflicts": conflicts,
        "result": (
            f"发现过敏风险：{json.dumps(conflicts, ensure_ascii=False)}"
            if conflicts else "未发现过敏冲突"
        ),
    }


async def _check_duplicate(drug_list: list[dict]) -> dict:
    """
    重复用药检查：同一处方中是否有同类药。

    目前按药品名分组，只能发现同名药重复开具；
    不同药名的同类药（如布洛芬与对乙酰氨基酚）不会被识别。
    """
    # 药品名 -> 出现次数对应的名称列表，长度 > 1 即为重复
    categories = {}
    for d in drug_list:
        name = d.get("name", "")
        if name:
            categories.setdefault(name, []).append(name)
    duplicates = {k: v for k, v in categories.items() if len(v) > 1}
    return {
        "type": "duplicate",
        "duplicates": duplicates,
        "result": f"重复用药：{duplicates}" if duplicates else "未发现重复用药",
    }


EMPTY_ANSWER = "未能从您的描述中识别出具体药品信息，请提供药品名称、剂量等详细信息。"


async def collect_prescription_checks(
    question: str, llm: BaseChatModel, embedding_model: Embeddings,
    milvus_client: MilvusClient, neo4j_driver: AsyncDriver,
    doc_options: DocRetrievalOptions | None = None,
) -> Evidence:
    """线上与评估共用解析、并发校验及失败状态，不把失败解释为安全。"""
    evidence = Evidence(question)
    prescription, status = await observe_source(_parse_prescription(question, llm))
    evidence.sources["prescription_parse"] = status.to_dict()
    drugs = (prescription or {}).get("drugs", [])
    if not drugs:
        return evidence
    drug_names = [d["name"] for d in drugs if d.get("name")]
    allergies = (prescription.get("patient_info") or {}).get("allergies", []) or []
    tasks = {
        "interaction": _check_interaction(drug_names, neo4j_driver, llm),
        "allergy": _check_allergy(drug_names, allergies, neo4j_driver, llm),
        "duplicate": _check_duplicate(drugs),
    }
    for drug in drugs:
        if drug.get("name"):
            tasks[f"dosage_{len(tasks)}"] = _check_dosage(
                drug["name"], drug.get("dosage"), embedding_model, milvus_client, doc_options,
            )
    checks = await asyncio.gather(*(observe_source(call) for call in tasks.values()))
    results = []
    for name, (value, status) in zip(tasks, checks):
        evidence.sources[name] = status.to_dict()
        if status.status in {"ok", "partial"}:
            evidence.used_sources.append(name)
            evidence.contexts.append(json.dumps(value, ensure_ascii=False, default=str))
        results.append(value if status.status != "failed" else {
            "type": name, "status": "failed", "issues": status.issues,
        })
    evidence.prompt_kwargs = {
        "prescription": json.dumps(prescription, ensure_ascii=False, indent=2),
        "check_results": json.dumps(results, ensure_ascii=False, indent=2, default=str),
    }
    return evidence


async def write_prescription_report(evidence: Evidence, llm: BaseChatModel) -> str:
    if not evidence.contexts:
        return EMPTY_ANSWER
    prompt = PRESCRIPTION_REPORT_PROMPT.format(**evidence.prompt_kwargs)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.text


async def review_prescription(
    question: str, llm: BaseChatModel, embedding_model: Embeddings,
    milvus_client: MilvusClient, neo4j_driver: AsyncDriver,
    doc_options: DocRetrievalOptions | None = None,
) -> str:
    evidence = await collect_prescription_checks(question, llm, embedding_model,
                                                 milvus_client, neo4j_driver, doc_options)
    return await write_prescription_report(evidence, llm)
