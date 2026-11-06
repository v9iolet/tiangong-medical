"""同集合数据隔离：用 chunk_strategy 区分原始分块与父子/语义变体；变体主键带专用前缀。"""
import json

VARIANT_PREFIX = 'ragvariant:'
ORIGINAL_FILTER = 'chunk_strategy == "original"'
# 写进集合 schema 的分块元数据字段（非空）及其 VARCHAR 长度
VARIANT_FIELDS = {'chunk_strategy': 32, 'parent_id': 256, 'parent_text': 65535, 'source_id': 256}


def original_chunk_metadata(chunk_id: str) -> dict:
    """原始分块以自身为来源；没有父块，不虚构父子关系。"""
    if not chunk_id or chunk_id.startswith(VARIANT_PREFIX):
        raise ValueError('原始分块 ID 不能为空或使用派生记录前缀')
    return dict(chunk_strategy='original', parent_id='', parent_text='', source_id=chunk_id)


def strategy_filter(strategy: str) -> str:
    return f'chunk_strategy == {json.dumps(strategy)}'
