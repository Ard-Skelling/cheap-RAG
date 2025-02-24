from os import getenv


# local module
from configs.config_v2.config_cls import (
    FileConvertConfig,
    OcrConfig,
    EmbeddingConfig,
    LlmConfig,
    WorkerConfig,
    ChunkingConfig,
    InsertPreprocessingConfig
)


FILE_CONVERT_CONFIG = FileConvertConfig()


OCR_CONFIG = OcrConfig(
    host=getenv('OCR_HOST') or '192.168.210.128',
    port=getenv('OCR_PORT') or '42018'
)


EMBEDDING_CONFIG = EmbeddingConfig(
    host=getenv('EMB_HOST') or '192.168.210.128',
    port=getenv('EMB_PORT') or '19998'
)


LLM_CONFIG = LlmConfig(
    host=getenv('LLM_HOST') or '192.168.210.117',
    port=getenv('LLM_PORT') or '1060'
)


WORKER_CONFIG = WorkerConfig()


CHUNKING_CONFIG = ChunkingConfig()


INSERT_PRE_CONFIG = InsertPreprocessingConfig()