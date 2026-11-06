"""从 Neo4j Symptom 节点初始化 symptoms_index，供问诊症状语义匹配使用。

用法：python -m scripts.init_symptom_index
向量模型统一读取 src.core.config；重复执行按症状名称 upsert。
"""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neo4j import AsyncGraphDatabase
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from src.core.config import get_embedding_model, get_settings

settings = get_settings()
COLLECTION_NAME = "symptoms_index"
MILVUS_ALIAS = "symptom_init"
BATCH_SIZE = 20


def ensure_symptom_collection(alias: str) -> Collection:
    """集合名称与 semantic_match_symptoms 的查询目标一致。"""
    if utility.has_collection(COLLECTION_NAME, using=alias):
        collection = Collection(COLLECTION_NAME, using=alias)
        dimension = next(field.params['dim'] for field in collection.schema.fields if field.name == 'embedding')
        if int(dimension) != settings.EMBEDDING_DIMENSIONS:
            raise ValueError('症状索引维度与当前向量配置不一致')
    else:
        schema = CollectionSchema([
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSIONS),
        ], description="Neo4j Symptom nodes vector index")
        collection = Collection(COLLECTION_NAME, schema=schema, using=alias)
        collection.create_index(field_name="embedding", index_params={
            "metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128},
        })
    collection.load()
    return collection


async def fetch_all_symptoms(neo4j_driver) -> list[str]:
    async with neo4j_driver.session() as session:
        result = await session.run("MATCH (s:Symptom) RETURN s.name AS name")
        records = await result.data()
    return sorted({record['name'] for record in records if record['name']})


async def build_symptom_index():
    connections.connect(alias=MILVUS_ALIAS, host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    try:
        collection = ensure_symptom_collection(MILVUS_ALIAS)
        driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        try:
            names = await fetch_all_symptoms(driver)
        finally:
            await driver.close()
        if not names:
            print('[WARN] Neo4j 中没有 Symptom 节点，请先导入医疗数据。')
            return
        embedding_model = get_embedding_model()
        data = []
        for offset in range(0, len(names), BATCH_SIZE):
            batch = names[offset:offset + BATCH_SIZE]
            vectors = await embedding_model.aembed_documents(batch)
            if len(vectors) != len(batch) or any(len(v) != settings.EMBEDDING_DIMENSIONS for v in vectors):
                raise ValueError('症状向量数量或维度不一致，尚未写入数据')
            data.extend(dict(id=name, name=name, embedding=vector) for name, vector in zip(batch, vectors))
            print(f'[INFO] 已向量化 {min(offset + BATCH_SIZE, len(names))}/{len(names)}')
        collection.upsert(data)
        collection.flush()
        print(f'[INFO] 症状索引构建完成，共写入 {len(data)} 条记录。')
    finally:
        connections.disconnect(alias=MILVUS_ALIAS)


if __name__ == '__main__':
    asyncio.run(build_symptom_index())
