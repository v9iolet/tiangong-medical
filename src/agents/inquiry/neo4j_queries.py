# src/agents/inquiry/neo4j_queries.py
# Neo4j 图谱查询封装：把 Cypher 查询结果转换成问诊流程使用的 CandidateDisease。
# 典型调用顺序：用户已确认症状找候选疾病 → 从候选疾病补充完整症状 → 根据详情生成待追问症状。
# 图谱关系方向与 scripts/init_neo4j.py 中的导入定义一致：
#   Disease --HAS_SYMPTOM--> Symptom      疾病具有哪些症状
#   Disease --BELONGS_TO--> Department    疾病对应哪些科室
#   Disease --NEED_CHECK--> Check         疾病建议做哪些检查
#   Disease --ACOMPANY_WITH--> Disease    疾病可能伴随哪些并发症

from __future__ import annotations
from loguru import logger
from neo4j import AsyncDriver
from src.agents.inquiry.state import CandidateDisease


async def query_candidate_diseases(
    confirmed_symptoms: list[str],
    neo4j_driver: AsyncDriver,
    top_k: int = 10,
) -> list[CandidateDisease]:
    """
    根据已确认症状列表，从 Neo4j 查询候选疾病。
    按基础置信度（命中症状数 / 该疾病总症状数）降序排列，取 Top K。
    """
    # 没有已确认症状时无法筛选候选，直接返回，避免一次无意义的数据库查询。
    if not confirmed_symptoms:
        return []

    cypher = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name IN $confirmed_symptoms
    WITH d, collect(s.name) AS matched_symptoms, count(s) AS matched_count
    MATCH (d)-[:HAS_SYMPTOM]->(all_s:Symptom)
    WITH d, matched_symptoms, matched_count, count(all_s) AS total_symptoms
    ORDER BY toFloat(matched_count) / total_symptoms DESC
    LIMIT $top_k
    RETURN
        d.name AS disease,
        matched_symptoms,
        matched_count,
        total_symptoms,
        toFloat(matched_count) / total_symptoms AS base_confidence
    """

    # $confirmed_symptoms / $top_k 是参数占位符，通过 run 的关键字参数绑定，
    # 不把用户输入直接拼接进 Cypher 字符串。
    # driver 由调用方创建和关闭；这里仅创建短生命周期的 session，退出时自动关闭。
    async with neo4j_driver.session() as session:
        result = await session.run(
            cypher,
            confirmed_symptoms=confirmed_symptoms,
            top_k=top_k,
        )
        # run() 返回异步结果对象；data() 也需要 await，才能得到 list[dict]。
        # 在 session 关闭前读取完结果，后面只处理内存中的字典。
        records = await result.data()

    # 将数据库字段映射为领域模型，后续节点无需了解 Cypher 返回格式。
    candidates = []
    for r in records:
        candidates.append(CandidateDisease(
            name=r["disease"],
            # 原始基础评分保留四位小数；confidence 从该值开始，后续可另行加权。
            base_confidence=round(r["base_confidence"], 4),
            confidence=round(r["base_confidence"], 4),  # 初始值，后续加权调整
            matched_symptoms=r["matched_symptoms"],
            # 第一阶段只查用于排名的信息，完整详情在第二阶段批量补充。
            all_symptoms=[],   # 由 enrich_candidate_details 补充
            department="",
            checks=[],
            complications=[],
        ))

    logger.debug(f"Neo4j 候选疾病: {[c.name for c in candidates]}")
    return candidates


async def enrich_candidate_details(
    candidates: list[CandidateDisease],
    neo4j_driver: AsyncDriver,
) -> list[CandidateDisease]:
    """
    补充候选疾病的完整信息：全部症状、建议科室、建议检查、并发症。
    在首次查到候选疾病后调用一次，后续轮次复用缓存在 state 里的数据。
    """
    if not candidates:
        return candidates

    # 一次传入所有候选疾病名，避免为每个疾病分别查询四类详情（N+1 查询）。
    disease_names = [c.name for c in candidates]

    symptoms_cypher = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(s.name) AS symptoms
    """

    dept_cypher = """
    MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department)
    WHERE d.name IN $names
    RETURN d.name AS disease, dept.name AS department
    """

    check_cypher = """
    MATCH (d:Disease)-[:NEED_CHECK]->(c:Check)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(c.name) AS checks
    """

    comp_cypher = """
    MATCH (d:Disease)-[:ACOMPANY_WITH]->(comp:Disease)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(comp.name) AS complications
    """

    # 分开查询四类关系，避免在一个查询中同时展开多种一对多关系，产生组合重复。
    # 这里按顺序执行四个查询，不是并行查询。
    # 两层 await 各有作用：内层等待 run 返回结果对象，外层等待 data 读取结果列表。
    # 等价写法：result = await session.run(...); rows = await result.data()。
    async with neo4j_driver.session() as session:
        symptoms_result = await (await session.run(symptoms_cypher, names=disease_names)).data()
        dept_result     = await (await session.run(dept_cypher,     names=disease_names)).data()
        check_result    = await (await session.run(check_cypher,    names=disease_names)).data()
        comp_result     = await (await session.run(comp_cypher,     names=disease_names)).data()

    # 将结果列表转换为“疾病名 → 详情”的字典，按名称关联，不依赖查询返回顺序。
    # 若一个疾病对应多个科室，dept_map 的同名键会被后面的行覆盖，只保留一个科室。
    symptoms_map    = {r["disease"]: r["symptoms"]      for r in symptoms_result}
    dept_map        = {r["disease"]: r["department"]    for r in dept_result}
    check_map       = {r["disease"]: r["checks"]        for r in check_result}
    comp_map        = {r["disease"]: r["complications"] for r in comp_result}

    # 上面的 MATCH 不会为缺失关系的疾病返回一行，因此用 get 提供空列表/空字符串。
    # 直接修改原 CandidateDisease 对象，并返回同一个列表；这里不重新计算评分。
    for c in candidates:
        c.all_symptoms  = symptoms_map.get(c.name, [])
        c.department    = dept_map.get(c.name, "")
        c.checks        = check_map.get(c.name, [])
        c.complications = comp_map.get(c.name, [])

    return candidates


async def get_pending_symptoms(
    candidates: list[CandidateDisease],
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    asked_symptoms: list[str],
) -> list[tuple[str, int]]:
    """
    从每个候选疾病的所有症状集合中，提取尚未确认/否认/追问过的症状。
    返回 [(症状名, 该症状出现在几个候选疾病中)]，选出症状区分度高的疾病，按出现次数升序排列。
    出现次数越少 = 区分度越高，下一轮优先追问用户该症状。
    """
    # 虽然函数声明为 async，但这里没有数据库调用或 await，只处理候选详情的内存数据。
    # 调用前应先补充 all_symptoms，否则默认的空列表无法生成待追问症状。
    # set 去重，| 取并集：确认、否认或已经问过的症状都不再追问。
    already_known = (set(confirmed_symptoms) |
                     set(denied_symptoms) |
                     set(asked_symptoms)
                     )
    # key 是症状名，value 是它在候选症状列表中出现的次数。
    # 假设各疾病的 all_symptoms 没有重复项，此次数才等于“包含它的候选疾病数量”。
    symptom_count: dict[str, int] = {}
    for c in candidates:
        for s in c.all_symptoms:
            if s not in already_known:
                # 首次出现从 0 开始计数，后续遇到同一症状继续累加。
                symptom_count[s] = symptom_count.get(s, 0) + 1

    # items() 得到 (症状名, 次数)，x[1] 按次数排序，而不是按症状名排序。
    # 例如 [("咳嗽", 3), ("皮疹", 1)] 会排成 [("皮疹", 1), ("咳嗽", 3)]。
    sorted_symptoms = sorted(symptom_count.items(), key=lambda x: x[1])
    return sorted_symptoms