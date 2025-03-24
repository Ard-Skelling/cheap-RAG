import re
from os import getenv
from typing import Optional, List
from pydantic_settings import BaseSettings


# local module
from configs.config_cls import (
    LocalEmbeddingConfig,
    OcrConfig,
    LlmConfig,
    WorkerConfig,
    ChunkingConfig,
    InsertPreprocessingConfig,
    QueryConfig
)


class IgnoreRule(BaseSettings):
    # Regex to identify the starting block to dropout
    name: str
    field: str
    regex: str
    re_args: tuple = (re.I, )
    # How many blocks to drop since then
    successor: int = 0
    succ_type: str = 'text'


class PaperChunkingConfig(ChunkingConfig):
    # Aggregated chunk token threshold
    agg_size: int = 3000    
    # Overlap tokens between aggregated chunks
    agg_overlap: int = 500   
    # Atomic chunk token threshold 
    atom_size: int = 200    
    # Show table markdown + url in aggregated chunk or url only
    show_table_md: bool = False    
    # Show image markdown + url in aggregated chunk or url only
    show_image_md: bool = False
    build_graph: bool = False
    build_summary: bool = False
    build_text_qa: bool = False
    build_table_qa: bool = False
    build_image_qa: bool = False
    # Layout block headers to dropout
    # ignores: Optional[List[IgnoreRule]] = None


class PaperInsertPreprocessingConfig(InsertPreprocessingConfig):
    ...


LOCAL_EMBEDDING_CONFIG = LocalEmbeddingConfig(
    emb_type='cuda',
    model_dir='/root/nlp/model-params/intfloat/multilingual-e5-large-instruct',
    model_path='/root/nlp/model-params/intfloat/multilingual-e5-large-instruct/onnx/model.onnx'
)


QUERY_EMBEDDING_CONFIG = LocalEmbeddingConfig(
    emb_type='cpu',
    model_dir='/root/nlp/model-params/intfloat/multilingual-e5-large-instruct/onnx',
    model_path='/root/nlp/model-params/intfloat/multilingual-e5-large-instruct/onnx/model.onnx'
)


OCR_CONFIG = OcrConfig(
    base_url='http://127.0.0.1:48308/v1/ocr',
    timeout=3600,
    sema_process=4
)


LLM_CONFIG = LlmConfig(
    llm_type='openai',
    base_url='https://api.aimlapi.com/v1',
    timeout=1800,
    api='/v1/chat/completions',
    semaphore=16,
    model='gpt-4o-mini',
    token=getenv('AIMLAPI_KEY'),
    temperature=0.5,
    max_tokens=8192
)


WORKER_CONFIG = WorkerConfig(
    num_workers=4,
    num_semaphore=16,
    timeout=9999,
    delete_old=True
)


CHUNKING_CONFIG = PaperChunkingConfig(
    agg_size=2000,
    agg_overlap=400,
    atom_size=200,    # Atomic chunk tokens
    show_table_md=False,
    # ignores=[
    #     IgnoreRule(
    #         field='text', 
    #         regex='^references$', 
    #         successor=1,
    #         succ_type='text'
    #     )
    # ]        
)


IPP_CONFIG = PaperInsertPreprocessingConfig()


QUERY_CONFIG = QueryConfig(
    semaphore=200
)

