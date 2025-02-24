import multiprocessing
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_PATH = Path(__file__).parent.parent.parent


# 初始化缓存目录
CACHE = BASE_PATH.joinpath('.cache')

# 初始化原始文件存储目录
RAW_FILE_CACHE = CACHE.joinpath('v2_raw_file')
RAW_FILE_CACHE.mkdir(parents=True, exist_ok=True)


# 初始化文件转换存储目录
FILE_CONVERT_CACHE = CACHE.joinpath('v2_file_convert')
FILE_CONVERT_CACHE.mkdir(parents=True, exist_ok=True)


# 初始化ocr结果缓存目录
OCR_CACHE = CACHE.joinpath('v2_ocrs')
OCR_CACHE.mkdir(parents=True, exist_ok=True)


class FileConvertConfig(BaseSettings):
    num_workers: int = max(int(multiprocessing.cpu_count() / 8), 1)
    # TODO：任务队列中的文件数量，超过的将会持久化到硬盘，之后再读取
    max_queue_size: int = 100
    raw_cache: Path = RAW_FILE_CACHE
    convert_cache: Path = FILE_CONVERT_CACHE
    timout:float = 300


class OcrConfig(BaseSettings):
    host: str
    port: str
    timeout: float = 99999
    use_minio: bool = True
    predict_api: str = '/predict'
    download_api: str = '/download'
    sema_predict: int = 16    # 提交ocr的并发量
    sema_download: int = 128    # 下载图片并发量
    ocr_cache: Path = OCR_CACHE


class WorkerConfig(BaseSettings):
    num_workers: int = max(int(multiprocessing.cpu_count() / 8), 1)
    num_semaphore:int = 16    # 全局并发量，并控制写工作流的任务队列上限
    raw_cache: Path = RAW_FILE_CACHE
    convert_cache: Path = FILE_CONVERT_CACHE
    timeout: float = 99999


class EmbeddingConfig(BaseSettings):
    host: str
    port: str
    timeout: float = 1800
    api: str = '/v1/embeddings'
    semaphore: int = 16
    model:str = "rgzn_bge_model"
    batch_size: int = 128


class LlmConfig(BaseSettings):
    host: str
    port: str
    timeout: float = 99999
    api: str = '/v1/chat/completions'
    semaphore: int = 16
    model: str = 'llm_72b'


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