from os import getenv


# local module
from configs.config_cls import (
    LoggerConfig,
    MilvusConfig,
    MinioConfig,
    ESConfig,
    FileConvertConfig,
    OcrConfig,
    EmbeddingConfig,
    LocalEmbeddingConfig,
    LlmConfig,
    WorkerConfig,
    ChunkingConfig,
    InsertPreprocessingConfig
)


MACHINE_ID = 1


LOGGER_CONFIG = LoggerConfig(
    file_name='cheap_rag.log',
    level='INFO',
    when='midnight',
    interval=1,
    backup_count=90,
    delay=False,
    utc=True
)


# storage config
MILVUS_CONFIG = MilvusConfig(
    uri='http://127.0.0.1:19530',
    auto_release_seconds=300
)


MINIO_CONFIG = MinioConfig(
    host='127.0.0.1',
    port=9000,
    bucket='file-meta',
    bucket_ocr='file-ocr',
    max_workers=32
)


ES_CONFIG = ESConfig(
    host='127.0.0.1',
    port=9200
)


# tools config
OCR_CONFIG = OcrConfig(
    base_url='http://127.0.0.1:20771',
    timeout=3600,
    sema_process=16
)


EMBEDDING_CONFIG = EmbeddingConfig(
    emb_type='openai',
    base_url='https://api.aimlapi.com/v1',
    timeout=1800,
    api='',
    semaphore=16,
    model="BAAI/bge-large-en-v1.5",
    batch_size=128
)


LOCAL_EMBEDDING_CONFIG = LocalEmbeddingConfig(
    emb_type='cuda',
    model_dir='/root/nlp/model-params/intfloat/multilingual-e5-large-instruct',
    model_path='/root/nlp/model-params/intfloat/multilingual-e5-large-instruct/onnx/model.onnx',
    batch_size=16
)


LLM_CONFIG = LlmConfig(
    base_url='https://api.aimlapi.com/v1',
    timeout=1800,
    api='/v1/chat/completions',
    semaphore=16,
    model='gpt-4o-mini',
    token=getenv('AIMLAPI_KEY')
)


# workflow config
FILE_CONVERT_CONFIG = FileConvertConfig()


WORKER_CONFIG = WorkerConfig()


CHUNKING_CONFIG = ChunkingConfig()


INSERT_PRE_CONFIG = InsertPreprocessingConfig()