from modules.storage.milvus_storage import (
    IDField,
    StrField,
    VectorField,
    CpuVectorIndexSetting
)


# Milvus collection design
FIELDS = [
    IDField(field_name='doc_id'),
    StrField(field_name='file_name', max_length=512),
    VectorField(field_name='vec', dim=1024)
]
INDEX_SETTINGS = [
    CpuVectorIndexSetting(field_name='vec')
]

# ElasticSearch index design
MAPPINGS = {
    "properties": {
        "doc_id": {"type": "keyword"},
        "text": {"type": "text"},
        "caption": {"type": "text"}, # image caption, table title
        "footnote": {"type": "text"},
        "file_name": {"type": "keyword"},
        "layout_index": {"type": "integer"},
        "atom_index": {"type": "integer"},
        "agg_index": {"type": "integer"},
        # chunk_type 支持 text, table, image, qa, outline, summary, atom_text, agg_text几种形式
        # atom_text 是原子化的文本片，是向量检索的最小单元
        # agg_text 是大文本片，是真正返回的片段
        "chunk_type": {"type": "keyword"},
        "url": {"type": "keyword"},
        "text_level": {"type": "integer"},
        "@timestamp": {"type": 'date'}
    }
}