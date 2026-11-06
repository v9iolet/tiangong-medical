# src/agents/inquiry/db_queries.py

# PostgreSQL 问诊数据封装：读取患者档案供问诊使用，保存问诊结果供后续查询。
# Patient / Consultation 是 ORM 模型，映射 patients / consultations 数据库表。
# PatientContext 是问诊状态的数据模型，不是表；这里将查询结果转换成它。
# Neo4j 查询模块负责疾病关系，本模块负责患者档案和问诊记录。

from __future__ import annotations
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from src.modules.medical.model import Patient, Consultation
from src.agents.inquiry.state import PatientContext


async def load_patient_context(
    patient_id: int | None,
    db: AsyncSession | None = None,
    recent_consultations_limit: int = 5,
) -> PatientContext:
    """
    从 PostgreSQL 加载患者上下文：基本信息 + 近期问诊记录。
    patient_id 为 None 时返回空上下文。
    db 为 None 时自动创建 session。
    """
    # 无患者 ID 时直接返回默认上下文，不创建 session，也不查询数据库。
    if patient_id is None:
        return PatientContext()

    # AsyncSessionLocal 是异步会话工厂；调用它才创建本次操作使用的 session。
    from src.infra.database import AsyncSessionLocal
    # 记录会话所有权：自己创建的自己关闭，外部传入的交由调用方管理。
    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()

    try:
        # 查患者基本信息
        # select 构造查询、where 按主键筛选；execute 才执行 SQL 并等待数据库返回。
        # 返回的是结果对象，不是 Patient 对象本身。
        patient_result = await db.execute(
            select(Patient).where(Patient.id == patient_id)
        )
        # scalar_one_or_none() 取唯一的 Patient 对象；无记录时为 None，多条则报错。
        # execute 返回缓冲结果，此处取数据不再需要 await，区别于 Neo4j 的 data()。
        patient = patient_result.scalar_one_or_none()
        if not patient:
            logger.warning(f"患者 ID {patient_id} 不存在")
            return PatientContext()

        # 解析既往病史和过敏史（数据库中用逗号分隔存储）
        # split(",") 按英文逗号拆分，strip() 去首尾空白，if h.strip() 过滤空项。
        # 例如 "疾病A, 疾病B,," → ["疾病A", "疾病B"]；None/空字符串返回 []。
        # 当前不处理中文逗号，也不对档案中的重复项去重。
        medical_history = (
            [h.strip() for h in patient.medical_history.split(",") if h.strip()]
            if patient.medical_history else []
        )
        allergy_history = (
            [a.strip() for a in patient.allergy_history.split(",") if a.strip()]
            if patient.allergy_history else []
        )

        # 查近期问诊记录（取最近 N 条）
        # where 只取该患者的记录；created_at 是从 ORM 公共基类继承的创建时间。
        # desc 按时间降序（最新在前），limit 在数据库端限制条数，默认最近 5 条。
        consultations_result = await db.execute(
            select(Consultation)
            .where(Consultation.patient_id == patient_id)
            .order_by(desc(Consultation.created_at))
            .limit(recent_consultations_limit)
        )
        # scalars() 提取每行的 Consultation 对象，all() 收集成列表；无记录时为 []。
        # 与上面只取一个患者不同，这里预期有多条问诊记录。
        consultations = consultations_result.scalars().all()

        # 把历史诊断结论加入 medical_history（用于置信度加权）
        # not in 按完整字符串相等判断重复，不是语义去重，也不拆分诊断文本。
        # 只合并到内存列表，不会修改 Patient 表原来的 medical_history 字段。
        for c in consultations:
            if c.diagnosis and c.diagnosis not in medical_history:
                medical_history.append(c.diagnosis)

        # 转换为问诊上下文，让后续流程无需操作 ORM 对象。
        # long_term_memories 未在此加载，保持模型默认空列表，需由其他流程补充。
        return PatientContext(
            patient_id=patient_id,
            age=patient.age,
            gender=patient.gender,
            allergy_history=allergy_history,
            medical_history=medical_history,
        )
    except Exception as e:
        # 查询或数据处理失败时降级为空上下文；返回值无法区分缺少档案与查询失败。
        # 排查原因需要结合日志；本分支不回滚外部 session 的事务。
        logger.error(f"加载患者上下文失败: {e}")
        return PatientContext()
    finally:
        # try 中即使提前 return，finally 仍会执行；只关闭自己创建的 session。
        if _own_session:
            await db.close()


async def save_consultation_record(
    patient_id: int | None,
    session_id: str,
    chief_complaint: str,
    diagnosis: str,
    department_name: str,
    urgency_level: str,
    db: AsyncSession | None = None,
) -> int | None:
    """
    将本次问诊结果保存到 PostgreSQL consultations 表。
    返回新记录的 ID，失败时返回 None。
    db 为 None 时（如从 Supervisor 工具链调用）自动创建 session。
    """
    from src.infra.database import AsyncSessionLocal

    # db 为 None 时自己创建 session，用完后关闭
    # db 是数据库会话，而 session_id 是业务对话 ID，二者不是同一个概念。
    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()

    try:
        # 查科室 ID（通过科室名查）
        from src.modules.medical.model import Department
        # 输入是科室名称，consultations 表保存的是科室外键 ID，因此先做名称到 ID 的转换。
        dept_result = await db.execute(
            select(Department).where(Department.name == department_name)
        )
        dept = dept_result.scalar_one_or_none()
        # 查不到科室时使用 None；当前模型的 department_id 允许为空。
        dept_id = dept.id if dept else None

        # 创建 ORM 对象，此时还没有写入数据库，也没有数据库生成的主键 ID。
        consultation = Consultation(
            patient_id=patient_id,  # 当前 ORM 为 nullable=False；传 None 会违反非空约束。
            department_id=dept_id,
            chief_complaint=chief_complaint,
            diagnosis=diagnosis,
            urgency_level=urgency_level,
            session_id=session_id,
        )
        # add() 将对象放入 session 的待写入集合，是同步方法，不需要 await。
        db.add(consultation)
        # commit() 先 flush 写入，再提交事务；即使 session 来自调用方，也会在这里提交。
        # 若该 session 中还有其他待写入对象，也会随当前事务一起提交。
        await db.commit()
        # refresh() 重新读取记录，更新数据库生成的字段，如 ID、创建时间。
        await db.refresh(consultation)
        logger.info(f"问诊记录已保存，ID={consultation.id}")
        return consultation.id
    except Exception as e:
        await db.rollback()
        logger.error(f"保存问诊记录失败: {e}")
        return None
    finally:
        # 和读取函数一样，只关闭本函数创建的 session，不关闭调用方传入的。
        if _own_session:
            await db.close()