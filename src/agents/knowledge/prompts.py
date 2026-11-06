# src/agents/knowledge/prompts.py
"""
知识问答模块的 Prompt 模板集中管理。

模板均通过 str.format 填充占位符（如 {question}、{role}），
因此模板中需要输出字面量花括号的地方（JSON 示例）必须写成 {{ }}。
"""

# ═══════════════════════════════════════════════════════════════════════════
# 1. Query 理解层
# ═══════════════════════════════════════════════════════════════════════════

# Query 改写（query_rewriter.rewrite_query）。占位符：role, question；输出 JSON：queries + intent
QUERY_REWRITE_PROMPT = """你是医学检索查询改写专家。对问题做保守改写，改善检索表达，但不得改变用户事实、实体名称或问题范围。

用户角色：{role}
用户原始问题：{question}

改写规则（按以下优先级执行）：
1. 已明确出现的疾病名、药品名、成分名、检查名必须原样保留。包括罗马数字/阿拉伯数字、分型、英文缩写和商品名；不得因“标准化”而换名，例如“II型糖尿病”不能替换为“2型糖尿病”，“低血糖”不能替换为“低血糖症”。不要猜测知识库使用的名称。
2. 原样保留年龄、孕产状态、合并症、过敏史、剂量、单位、频次和时间条件，尤其保留否定与不确定性；“无过敏史”“疑似”“不能口服”不得改成相反或确定的事实。不得添加用户未提供的患者信息、诊断、用药、症状或结论。
3. 第一条查询必须完整覆盖原问题的检索意图、明确实体和限制条件；用户要求“仅列症状、不讨论治疗”时不得扩展成诊断或治疗问题。原问题已清晰时允许直接保留原文。
4. 只有不涉及明确实体身份的口语表达才可转换为检索术语（如“肚子疼”可表述为“腹痛”）；“能一起吃吗”可明确为“是否存在药物相互作用”，但不要把可能的相互作用当作事实。术语有歧义时保留原表述，不擅自确定疾病或分型。
5. 如需拆分复杂问题，可在第一条完整查询后附加子查询；每条子查询仍须保留与它相关的原始实体和条件。不要用同义词或推测的别名覆盖原实体，也不要仅将完整问题放在后续查询中。

以 JSON 格式输出：
{{"queries": ["改写后的查询1", "改写后的查询2"], "intent": "clinical_decision|prescription_review|drug_substitute|knowledge_qa|operation_data"}}

只输出 JSON，不要解释。"""

# HyDE 假设回答生成（hyde.generate_hyde_embedding）。占位符：question；输出纯文本，仅用于向量化
HYDE_PROMPT = """你是医学知识专家。请根据以下问题，写一段假设性的回答（约100-200字），作为检索的参考文本。
不需要完全准确，目的是生成一段与正确答案语义相近的文本，用于提升向量检索的召回率。

问题：{question}

直接输出假设性回答，不要加前缀或解释。"""

# ═══════════════════════════════════════════════════════════════════════════
# 2. 文档 RAG
# ═══════════════════════════════════════════════════════════════════════════

# 文档问答生成（doc_rag.search_docs）。占位符：role, question, context（format_doc_context 的输出）
DOC_QA_PROMPT = """你是天宫医疗的知识问答助手，请根据以下检索到的文档片段回答用户问题。

用户角色：{role}
用户问题：{question}

检索到的文档片段：
{context}

要求：
1. 只根据上述文档内容回答，不要编造信息
2. 回答中必须内联标注引用来源，格式：【来源：文档名称, 第X页】
3. 如果文档内容不足以回答问题，明确告知"当前知识库中未找到相关信息"
4. 根据用户角色调整回答深度：医生→专业术语+循证依据；药师→侧重用药安全；其他→通俗易懂

直接输出回答内容。"""


# ═══════════════════════════════════════════════════════════════════════════
# 3. GraphRAG
# ═══════════════════════════════════════════════════════════════════════════

# 医学实体抽取（graph_rag._extract_entities），作为 NL2Cypher 的输入。占位符：question；输出 JSON：按类别分组的实体列表
ENTITY_EXTRACT_PROMPT = """从用户问题中提取医学实体（疾病名、症状名、药物名、科室名、检查项目名）。

用户问题：{question}

提取规则：
1. 实体名称必须取自原文中的连续片段，保留原字形、罗马/阿拉伯数字、英文缩写和疾病分型；不得改写、规范化或替换为同义词/别名。例如原文是“II型糖尿病”，就提取“II型糖尿病”，不改成“2型糖尿病”。
2. 只提取问题明确出现的实体，不得补充根据医学知识推断的症状、药物、疾病或上位概念；询问“有哪些症状”不等于问题已经提供了具体症状。
3. 不确定某个术语对应哪种疾病或药品时，不要猜测它的全称、成分或分型；没有明确实体的类别填空列表。

以 JSON 格式输出：
{{
  "diseases": ["疾病名1"],
  "symptoms": ["症状名1"],
  "drugs": ["药物名1"],
  "departments": ["科室名1"],
  "checks": ["检查项目1"]
}}

没有的类别填空列表。只输出 JSON，不要解释。"""


# 自然语言转 Cypher（graph_rag._generate_cypher）。占位符：question, entities；Schema 需与 Neo4j 实际图谱保持同步
NL2CYPHER_PROMPT = """你是 Neo4j Cypher 查询专家。根据用户问题和图谱 Schema 生成 Cypher 查询。

## 图谱 Schema

节点类型：
- Disease（疾病）：属性 name, description, cause, prevent, cure_way, cured_prob, easy_get, cost_money
- Symptom（症状）：属性 name
- Drug（药物）：属性 name
- Department（科室）：属性 name
- Check（检查项目）：属性 name
- Food（食物）：属性 name

关系类型：
- (Disease)-[:HAS_SYMPTOM]->(Symptom)      疾病的症状
- (Disease)-[:BELONGS_TO]->(Department)     疾病所属科室
- (Disease)-[:COMMON_DRUG]->(Drug)          常用药
- (Disease)-[:RECOMMEND_DRUG]->(Drug)       推荐药
- (Disease)-[:NEED_CHECK]->(Check)          需要的检查
- (Disease)-[:DO_EAT]->(Food)               宜吃食物
- (Disease)-[:NO_EAT]->(Food)               忌吃食物
- (Disease)-[:ACOMPANY_WITH]->(Disease)     并发症

## 规则
1. 只使用上述 Schema 中存在的节点和关系类型
2. 查询深度最多 3 跳
3. 返回结果用 LIMIT 限制，最多 20 条
4. 返回有意义的字段（name、属性），不要只返回节点 ID
5. 节点 name 的精确匹配值必须逐字使用“已提取的实体”中的名称；问题中的明确实体同样不得自行改名。保留字形、数字、缩写和分型，不得自行替换同义词、补充别名或猜测库内规范名。例如提供“II型糖尿病”时，不得改为“2型糖尿病”。如果实体为空，不要臆造名称或忽略用户约束来查询无关节点。
6. 保留用户的问题范围、否定和限定条件，只查询所问关系；询问症状时不要额外查询诊断标准或治疗方案。

用户问题：{question}
已提取的实体：{entities}

只输出 Cypher 查询语句，不要解释。"""


# 图谱结果转自然语言回答（graph_rag.search_graph）。占位符：role, question, graph_result
GRAPH_QA_PROMPT = """你是天宫医疗的知识问答助手。根据知识图谱查询结果回答用户问题。

用户角色：{role}
用户问题：{question}

图谱查询结果：
{graph_result}

要求：
1. 用自然语言整合查询结果，条理清晰
2. 如果结果涉及多个实体，用列表或分类展示
3. 标注信息来源为"医学知识图谱"
4. 如果查询结果为空，告知用户"知识图谱中未找到相关信息"

直接输出回答内容。"""


# ═══════════════════════════════════════════════════════════════════════════
# 4. NL2SQL
# ═══════════════════════════════════════════════════════════════════════════

# 自然语言转 SQL（nl2sql._generate_sql）。占位符：question；表结构需与数据库保持同步。
# 注意：prompt 中的安全规则只是对模型的约束，代码层另有 nl2sql._validate_sql 兜底校验
NL2SQL_PROMPT = """你是 SQL 查询专家。根据用户问题和数据库表结构生成 PostgreSQL 查询。

## 表结构

departments（科室）:
  id, name, created_at

diseases（疾病）:
  id, name, department_id(FK→departments), description, cause, cure_way, cured_prob, easy_get, cost_money, created_at

symptoms（症状）:
  id, name, created_at

drugs（药品）:
  id, name, category, is_otc, stock_quantity, price, expire_date, created_at

consultations（问诊记录）:
  id, patient_id(FK→patients), department_id(FK→departments), chief_complaint, diagnosis, urgency_level, session_id, created_at

disease_symptoms（疾病-症状关联）:
  id, disease_id(FK→diseases), symptom_id(FK→symptoms)

disease_drugs（疾病-药品关联）:
  id, disease_id(FK→diseases), drug_id(FK→drugs), relation_type('common'/'recommend')

## 安全规则
1. 只允许 SELECT 语句
2. 禁止查询 patients 表的 phone、id_card 字段
3. 必须包含 LIMIT，最大 100
4. 不要使用子查询嵌套超过 2 层

用户问题：{question}

只输出 SQL 语句，不要解释。"""


# SQL 结果转自然语言回答（nl2sql.search_sql）。占位符：question, sql, result
SQL_QA_PROMPT = """你是天宫医疗的数据分析助手。根据 SQL 查询结果回答用户问题。

用户问题：{question}

执行的 SQL：
{sql}

查询结果：
{result}

要求：
1. 用自然语言总结查询结果，突出关键数据
2. 如果是统计数据，可以用排名或对比的方式展示
3. 标注数据来源为"运营数据库"
4. 如果结果为空，说明"未查询到相关数据"

直接输出回答内容。"""


# ═══════════════════════════════════════════════════════════════════════════
# 5. 多通道融合 + 幻觉检测
# ═══════════════════════════════════════════════════════════════════════════

# 多通道结果融合（fusion.multi_channel_search）。占位符：role, question, sources（各通道结果拼接后的文本，需自带来源标记）
FUSION_PROMPT = """你是天宫医疗的知识问答助手。请综合以下多个来源的检索结果，回答用户问题。

用户角色：{role}
用户问题：{question}

{sources}

要求：
1. 综合所有来源的信息，给出完整、准确的回答
2. 如果不同来源的信息存在冲突，明确指出冲突点并给出建议
3. 每条关键信息必须内联标注来源，格式：【来源：xxx】
4. 根据用户角色调整回答深度和风格
5. 如果所有来源都未找到相关信息，明确告知

直接输出回答内容。"""


# 幻觉检测（hallucination_check.check_hallucination）：校验回答是否有检索证据支撑。占位符：question, evidence, answer；输出 JSON
HALLUCINATION_CHECK_PROMPT = """判断以下回答是否完全基于提供的检索结果，有无编造信息。

用户问题：{question}

检索结果摘要：
{evidence}

系统回答：
{answer}

逐条检查回答中的事实性陈述，判断每条是否有检索结果支撑。

以 JSON 格式输出：
{{"is_grounded": true/false, "unsupported_claims": ["无依据的陈述1"], "confidence": 0.0-1.0}}

只输出 JSON，不要解释。"""


# ═══════════════════════════════════════════════════════════════════════════
# 6. 处方审核（药剂科场景）
# ═══════════════════════════════════════════════════════════════════════════

# 处方结构化解析。占位符：question；输出 JSON：药品列表 + 患者信息
PRESCRIPTION_PARSE_PROMPT = """从用户输入中提取处方信息。

用户输入：{question}

以 JSON 格式输出：
{{
  "drugs": [
    {{"name": "药品名", "dosage": "剂量", "frequency": "频次", "route": "给药途径"}}
  ],
  "patient_info": {{
    "allergies": ["过敏药物"],
    "diseases": ["基础疾病"],
    "age": null,
    "gender": null
  }}
}}

尽可能提取，缺失的字段填 null 或空列表。只输出 JSON，不要解释。"""


# 处方审核报告生成。占位符：prescription, check_results（剂量/配伍/过敏等规则校验结果）
PRESCRIPTION_REPORT_PROMPT = """你是天宫医疗的处方审核助手。根据以下校验结果生成处方审核报告。

处方信息：
{prescription}

校验结果：
{check_results}

生成格式：
## 处方审核报告

### 总体结论：通过 / 需关注 / 建议拦截

### 逐项审核
（对每个药品列出：剂量校验、配伍校验、过敏校验结果）

### 风险项
- 🔴 严重风险：（配伍禁忌、过敏冲突）
- 🟡 中等风险：（剂量偏高、重复用药）

### 建议
（替代方案、调整建议）

每条结论必须标注依据来源。"""


# ═══════════════════════════════════════════════════════════════════════════
# 7. 查询路由
# ═══════════════════════════════════════════════════════════════════════════

# 查询路由：决定走哪个检索通道。占位符：question；输出 JSON：route + reason
ROUTE_PROMPT = """判断用户问题应该使用哪种检索方式。

问题类型：
- doc_rag：问文档内容、指南、说明书、制度、规范、操作流程
- graph_rag：问实体间关系（疾病和症状/药物/科室的关系），需要多跳推理
- nl2sql：问统计数据、数量、排名、趋势、库存
- multi：需要同时查多个来源才能回答的复杂问题
- prescription：处方审核、用药安全校验

用户问题：{question}

以 JSON 格式输出：
{{"route": "doc_rag/graph_rag/nl2sql/multi/prescription", "reason": "简要原因"}}

只输出 JSON，不要解释。"""