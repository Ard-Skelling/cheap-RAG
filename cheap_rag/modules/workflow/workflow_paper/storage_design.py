from modules.storage.milvus_storage import (
    IDField,
    StrField,
    IntField,
    VectorField,
    CpuVectorIndexSetting
)


# Milvus collection design
FIELDS = [
    IDField(field_name='atom_id', max_length=32),
    IntField(field_name='agg_index'),
    StrField(field_name='file_name', max_length=512),
    VectorField(field_name='vec', dim=1024)
]
INDEX_SETTINGS = [
    CpuVectorIndexSetting(field_name='vec')
]


# ElasticSearch index design
ATOM_MAPPINGS = {
    "properties": {
        "atom_id": {"type": "keyword"},
        "text": {"type": "text"},
        "agg_index": {"type": "integer"},
        "file_name": {"type": "keyword"},
        "@timestamp": {"type": 'date'}
    }
}

AGG_MAPPINGS = {
    "properties": {
        "agg_id": {"type": "keyword"},
        "text": {"type": "text"},
        "file_name": {"type": "keyword"},
        "agg_index": {"type": "integer"},
        "page_index": {"type": "integer"},
        "chunk_type": {"type": "keyword"},
        "@timestamp": {"type": 'date'}
    }
}

RAW_MAPPINGS = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "agg_index": {"type": "integer"},
        "chunk_index": {"type": "integer"},
        "page_index": {"type": "integer"},
        "text": {"type": "text"},
        "caption": {"type": "text"}, # image caption, table title
        "footnote": {"type": "text"},
        "file_name": {"type": "keyword"},
        "chunk_type": {"type": "keyword"},
        "url": {"type": "keyword"},
        "text_level": {"type": "integer"},
        "@timestamp": {"type": 'date'}
    }
}


ES_SETTINGS = [
    {'suffix': 'atom', 'mappings': ATOM_MAPPINGS},
    {'suffix': 'agg', 'mappings': AGG_MAPPINGS},
    {'suffix': 'raw', 'mappings': RAW_MAPPINGS}
]


