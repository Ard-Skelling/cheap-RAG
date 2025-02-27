import multiprocessing
from typing import Optional, Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


BASE_PATH = Path(__file__).parent.parent


# 初始化缓存目录
CACHE = BASE_PATH.joinpath('.cache')

# 初始化原始文件存储目录
RAW_FILE_CACHE = CACHE.joinpath('raw_file')
RAW_FILE_CACHE.mkdir(parents=True, exist_ok=True)


# 初始化文件转换存储目录
FILE_CONVERT_CACHE = CACHE.joinpath('file_convert')
FILE_CONVERT_CACHE.mkdir(parents=True, exist_ok=True)


# 初始化ocr结果缓存目录
OCR_CACHE = CACHE.joinpath('ocrs')
OCR_CACHE.mkdir(parents=True, exist_ok=True)


# Logger config
class LoggerConfig(BaseSettings):
    file_name: str = 'cheap_rag.log'
    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
    when: str = 'midnight'    # S-Seconds, M-Minutes, H-Hours, D-Days, midnight, W{0-6}-certain day
    interval: int = 1
    backup_count: int = 90
    delay: bool = False
    utc: bool = True
    

# Storage config
class MilvusConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='milvus_')
    
    uri: str = 'http://localhost:19530'
    user: SecretStr = ''
    password: SecretStr = ''
    token: SecretStr = ''
    auto_release_seconds: float = 300.
    default_collection: Optional[str] = None


class MinioConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='minio_')


    host: str
    port: int
    ak: SecretStr
    sk: SecretStr
    bucket: str
    bucket_ocr: str
    max_workers: int = 32


class ESConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='es_')

    host: str
    port: int
    user: SecretStr
    pwd: SecretStr


# File format config
class FileConvertConfig(BaseSettings):
    num_workers: int = max(int(multiprocessing.cpu_count() / 8), 1)
    # TODO：任务队列中的文件数量，超过的将会持久化到硬盘，之后再读取
    max_queue_size: int = 100
    raw_cache: Path = RAW_FILE_CACHE
    convert_cache: Path = FILE_CONVERT_CACHE
    timout:float = 300


class OcrConfig(BaseSettings):
    base_url: str
    timeout: float = 3600
    sema_process: int = 16    # 提交ocr的并发量
    ocr_cache: Path = OCR_CACHE


class WorkerConfig(BaseSettings):
    num_workers: int = max(int(multiprocessing.cpu_count() / 8), 1)
    num_semaphore:int = 16    # 全局并发量，并控制写工作流的任务队列上限
    raw_cache: Path = RAW_FILE_CACHE
    convert_cache: Path = FILE_CONVERT_CACHE
    timeout: float = 99999


class EmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='emb_')

    emb_type: Literal['rest', 'openai'] = 'openai'
    base_url: str
    timeout: float = 1800
    api: str = '/v1/embeddings'
    semaphore: int = 16
    model:str = ''
    batch_size: int = 128
    token: SecretStr = ''


class LocalEmbeddingConfig(BaseSettings):
    emb_type: Literal['ONNX', 'TensorRT']
    model_dir: str
    model_path: str = ''


class LlmConfig(BaseSettings):
    base_url: str
    timeout: float = 1800
    api: str = '/v1/chat/completions'
    semaphore: int = 16
    model: str = 'gpt-4o-mini'


class ChunkingConfig(BaseSettings):
    agg_size: int = 3000    # 聚合的长文本长度阈值，是检索返回的大长片
    agg_overlap: int = 500    # 设定大长片之间的交叠区间长度
    atom_size: int = 200    # 原子文本的长度阈值，是召回时计算相似的对象
    agg_table_md: bool = True    # 聚合片段中是否加入表格的markdown
    agg_image_md: bool = False    # 聚合片段中是否加入图片的markdown
    qa_for_table: bool = True    # 对表格生成问题
    qa_for_image: bool = False    # 对图片生成问题


class InsertPreprocessingConfig(BaseSettings):
    ...