# ============================================================
# 医疗数据 SQLAlchemy Model 定义
#
# 通用设计：
# - 继承 BaseModel，自带 id / created_at / updated_at
# - 字段命名用英文，comment 写中文业务含义
# - 长文本字段（描述、病因等）用 Text，结构化字段用 String/Integer
# - 关联关系通过外键维护，不在 ORM 层定义 relationship（保持简单）
# ============================================================

from sqlalchemy import String, Text, Integer, Float, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from src.core.base_model import BaseModel

# ----------------------------------------------------------
# 科室表
# 来源：medical.json 中的 departments 字段
# 用途：分诊推荐（症状→疾病→科室 图路径的终点）
# ----------------------------------------------------------
class Department(BaseModel):
    __tablename__ = "departments"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, comment="科室名称，如：内科、外科")
    description: Mapped[str | None] = mapped_column(Text, comment="科室简介")

# ----------------------------------------------------------
# 疾病表
# 来源：medical.json 中每条记录的疾病主体字段
# 用途：
#   1. 分诊病因分析的结构化查询
#   2. 疾病百科文本切片后入 Milvus 向量索引
# ----------------------------------------------------------
class Disease(BaseModel):
    __tablename__ = "diseases"

    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, comment="疾病名称")
    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id", ondelete="SET NULL"),
        comment="所属科室 ID"
    )
    description: Mapped[str | None] = mapped_column(Text, comment="疾病描述/定义")
    cause: Mapped[str | None] = mapped_column(Text, comment="病因")
    prevent: Mapped[str | None] = mapped_column(Text, comment="预防措施")
    cure_way: Mapped[str | None] = mapped_column(Text, comment="治疗方式，多种方式用逗号分隔")
    cure_lasttime: Mapped[str | None] = mapped_column(String(200), comment="治疗周期，如：1-2个月")
    cured_prob: Mapped[str | None] = mapped_column(String(200), comment="治愈概率，如：95%")
    easy_get: Mapped[str | None] = mapped_column(Text, comment="易感人群描述")
    cost_money: Mapped[str | None] = mapped_column(String(200), comment="参考费用区间")

    __table_args__ = (
        Index("ix_diseases_name", "name"),
    )

# ----------------------------------------------------------
# 症状表
# 来源：medical.json 中 symptom 字段拆分
# 用途：症状结构化提取后的实体匹配
# ----------------------------------------------------------
class Symptom(BaseModel):
    __tablename__ = "symptoms"

    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, comment="症状名称，如：发热、头痛")

    __table_args__ = (
        Index("ix_symptoms_name", "name"),
    )

# ----------------------------------------------------------
# 疾病-症状关联表（多对多）
# 来源：medical.json 中 symptom 字段
# 用途：症状→疾病推断的结构化查询（同时也会导入 Neo4j 作为图关系）
# ----------------------------------------------------------
class DiseaseSymptom(BaseModel):
    __tablename__ = "disease_symptoms"

    disease_id: Mapped[int] = mapped_column(
        ForeignKey("diseases.id", ondelete="CASCADE"),
        nullable=False,
        comment="疾病 ID"
    )
    symptom_id: Mapped[int] = mapped_column(
        ForeignKey("symptoms.id", ondelete="CASCADE"),
        nullable=False,
        comment="症状 ID"
    )

    __table_args__ = (
        Index("ix_disease_symptoms_disease", "disease_id"),
        Index("ix_disease_symptoms_symptom", "symptom_id"),
    )

# ----------------------------------------------------------
# 药品表
# 来源：medical.json 中 common_drug / recommand_drug 字段 + NMPA 数据
# 用途：
#   1. 药品信息查询（功能 3.1）
#   2. 药品说明书 RAG（功能 3.5）
#   3. NL2SQL 药品统计（功能 5.5）
# ----------------------------------------------------------
class Drug(BaseModel):
    __tablename__ = "drugs"

    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="药品通用名")
    alias: Mapped[str | None] = mapped_column(String(500), comment="别名/商品名，多个用逗号分隔", nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), comment="药品分类，如：抗生素、解热镇痛", nullable=True)
    manufacturer: Mapped[str | None] = mapped_column(String(200), comment="生产厂家", nullable=True)
    approval_number: Mapped[str | None] = mapped_column(String(100), comment="国药准字批准文号", nullable=True)
    is_otc: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否为非处方药（OTC）", nullable=True)
    stock_quantity: Mapped[int] = mapped_column(Integer, default=0, comment="库存数量（运营统计用）", nullable=True)
    price: Mapped[float | None] = mapped_column(Float, comment="参考零售价（元）", nullable=True)
    expire_date: Mapped[str | None] = mapped_column(String(50), comment="有效期至，格式：YYYY-MM-DD", nullable=True)

    __table_args__ = (
        Index("ix_drugs_name", "name"),
        Index("ix_drugs_category", "category"),
    )

# ----------------------------------------------------------
# 药品详情表（与 drugs 一对一）
# 单独拆出来是因为这些字段都是长文本，查列表时不需要加载
# 用途：药品说明书问答 RAG 的原始文本来源（功能 3.5）
# ----------------------------------------------------------
class DrugDetail(BaseModel):
    __tablename__ = "drug_details"

    drug_id: Mapped[int] = mapped_column(
        ForeignKey("drugs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        comment="关联药品 ID"
    )
    indication: Mapped[str | None] = mapped_column(Text, comment="适应症/功能主治")
    usage_dosage: Mapped[str | None] = mapped_column(Text, comment="用法用量")
    adverse_reaction: Mapped[str | None] = mapped_column(Text, comment="不良反应")
    contraindication: Mapped[str | None] = mapped_column(Text, comment="禁忌症")
    precaution: Mapped[str | None] = mapped_column(Text, comment="注意事项")
    interaction: Mapped[str | None] = mapped_column(Text, comment="药物相互作用（文本描述，结构化关系在 Neo4j）")
    storage: Mapped[str | None] = mapped_column(Text, comment="储存条件")
    full_instruction: Mapped[str | None] = mapped_column(Text, comment="完整说明书原文（RAG 切片用）")

# ----------------------------------------------------------
# 疾病-药品关联表（多对多）
# 来源：medical.json 中 common_drug / recommand_drug 字段
# 用途：药物查询时关联疾病上下文；同时导入 Neo4j 作为图关系
# ----------------------------------------------------------
class DiseaseDrug(BaseModel):
    __tablename__ = "disease_drugs"

    disease_id: Mapped[int] = mapped_column(
        ForeignKey("diseases.id", ondelete="CASCADE"),
        nullable=False,
        comment="疾病 ID"
    )
    drug_id: Mapped[int] = mapped_column(
        ForeignKey("drugs.id", ondelete="CASCADE"),
        nullable=False,
        comment="药品 ID"
    )
    # common=常用药，recommend=推荐药，对应 medical.json 的两个字段
    relation_type: Mapped[str] = mapped_column(
        String(20), default="common", comment="关系类型：common=常用药 / recommend=推荐药"
    )

    __table_args__ = (
        Index("ix_disease_drugs_disease", "disease_id"),
        Index("ix_disease_drugs_drug", "drug_id"),
    )

# ----------------------------------------------------------
# 患者档案表
# 用途：
#   1. 患者历史档案查询
#   2. 患者群体分析 NL2SQL
#   3. 患者画像构建
# ----------------------------------------------------------
class Patient(BaseModel):
    __tablename__ = "patients"

    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="患者姓名")
    gender: Mapped[str | None] = mapped_column(String(10), comment="性别：男/女")
    age: Mapped[int | None] = mapped_column(Integer, comment="年龄")
    phone: Mapped[str | None] = mapped_column(String(20), comment="联系电话")
    id_card: Mapped[str | None] = mapped_column(String(50), comment="身份证号（脱敏存储）")
    allergy_history: Mapped[str | None] = mapped_column(Text, comment="过敏史，多种用逗号分隔")
    medical_history: Mapped[str | None] = mapped_column(Text, comment="既往病史")
    blood_type: Mapped[str | None] = mapped_column(String(10), comment="血型，如：A/B/O/AB")

    __table_args__ = (
        Index("ix_patients_name", "name"),
    )

# ----------------------------------------------------------
# 问诊记录表
# 用途：
#   1. 患者历史问诊语义检索（向量化后存 Milvus）
#   2. 报告趋势对比
#   3. 运营数据查询 NL2SQL
# ----------------------------------------------------------
class Consultation(BaseModel):
    __tablename__ = "consultations"

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        comment="患者 ID"
    )
    department_id: Mapped[int | None] = mapped_column(
        ForeignKey("departments.id", ondelete="SET NULL"),
        comment="就诊科室 ID"
    )
    chief_complaint: Mapped[str | None] = mapped_column(Text, comment="主诉（患者自述症状）")
    diagnosis: Mapped[str | None] = mapped_column(Text, comment="诊断结论")
    prescription: Mapped[str | None] = mapped_column(Text, comment="处方内容（JSON 字符串）")
    urgency_level: Mapped[str] = mapped_column(
        String(20), default="normal", comment="紧急程度：normal=普通 / urgent=较急 / emergency=紧急"
    )
    session_id: Mapped[str | None] = mapped_column(String(100), comment="关联的 Redis 会话 ID")
    milvus_doc_id: Mapped[str | None] = mapped_column(String(100), comment="向量化后在 Milvus 中的文档 ID")

    __table_args__ = (
        Index("ix_consultations_patient", "patient_id"),
        Index("ix_consultations_department", "department_id"),
    )
