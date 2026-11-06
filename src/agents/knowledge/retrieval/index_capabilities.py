"""同集合策略预检：集合存在、必要字段齐全，且所选策略已有入库数据。"""
from src.agents.knowledge.retrieval.options import DocRetrievalOptions
from src.agents.knowledge.retrieval.variant_schema import VARIANT_FIELDS, strategy_filter


def validate_doc_index(client, options: DocRetrievalOptions) -> dict:
    if not client.has_collection(options.collection_name):
        raise ValueError('knowledge_docs 不存在，请先导入知识文档')
    info = client.describe_collection(options.collection_name)
    fields = {field['name'] for field in info['fields']}
    if not ({'text', 'embedding', 'id'} | set(VARIANT_FIELDS)).issubset(fields):
        raise ValueError('集合缺少必要字段，请先运行 scripts/rebuild_knowledge_collection.py 重建集合')
    if options.use_hybrid and 'sparse_embedding' not in fields:
        raise ValueError('混合检索需要 sparse_embedding，请先运行 scripts/rebuild_knowledge_collection.py 重建集合')
    if options.chunk_strategy != 'original':
        counts = client.query(collection_name=options.collection_name,
                              filter=strategy_filter(options.chunk_strategy),
                              output_fields=['count(*)'], consistency_level='Strong')
        if not counts or int(counts[0]['count(*)']) == 0:
            raise ValueError('该策略还没有入库数据')
    return {'collection_name': options.collection_name}
