import numpy as np
from typing import Tuple, Literal
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator, Field


class DataPoint(BaseModel):
    model_config: ConfigDict = ConfigDict(extra='ignore')

    domain: str


# 按数据在工作流中的的生命周期，划分以下数据类
# 工作流中，可能在同一时间窗口同时存在多种数据类


class TaskMeta(DataPoint):
    file_name: str
    csid: str
    pdf_fp: Path = None


class Task(BaseModel):
    task_id: int
    task_meta: TaskMeta
    # 当前待执行的任务环节，对应Worker中的类方法名
    step: Literal['convert', 'ocr', 'chunking', 'insert']
    status: Literal['pending', 'processing', 'completed', 'failed']
    result: dict = None


class FileData(DataPoint):
    file_name: str = None
    file_base64: str = None


class OcrData(DataPoint):
    file_name: str = None
    data: dict


class GraphInfo(DataPoint):
    before_context: str = ''
    table_md: str
    after_context: str = ''
    file_name: str
    agg_index: int
    index: int
    atom_index: int
    url: str = ''
    chunk_type: str


class ChunkData(DataPoint):
    file_name: str
    chunk_type: str
    doc_id: str
    answer: str
    create_time: str = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


class Chunk(ChunkData):
    # 由于存在交叠文本块，某些块会归属于两个聚合块。
    agg_index: int = None
    # 文本块本身索引。全局级文本块索引为0。
    index: int = 0
    chunk_type: str = 'text'


    @field_validator('agg_index', mode='before')
    def set_agg_index(cls, v, values):
        if 'dual_agg_index' in values:
            return values['dual_agg_index'][0]
        return v
    

class AggChunk(ChunkData):
    # 聚合文本块自身索引
    agg_index: int = 0
    chunk_type: str = 'agg_text'


class AtomChunk(ChunkData):
    agg_index: int = 0
    index: int = 0
    atom_index: int = 0
    chunk_type: str = 'atom_text'


class TableChunk(AtomChunk):
    question: str = ''
    url: str
    chunk_type: str = 'table'


class ImageChunk(AtomChunk):
    url: str
    chunk_type: str = 'image'


class MilvusData(DataPoint):
    # 老版本的写入Milvus字段，未启用
    question: str
    answer: str
    vec: list
    slice_vec: list
    q_slice_vec: list
    file_name: str
    index: int
    type_: str = Field(alias='type')
    collection_name: str
    url: str
    parent_title: str
    title: str
    create_time: str


class MilvusDataV2(BaseModel):
    # V2版本的写入Milvus字段，仅保留文本片id、向量和文件名三个字段
    # 写入Milvus前应调用to_ndarray方法，将向量字段转为np.ndarry类型
    doc_id: str
    vec: list
    file_name: str

    def to_ndarray(self):
        # Milvus数据库只接受np.ndarray数据类型，需将浮点数列表转化为np.ndarry，并保持其它字段
        return {'doc_id': self.doc_id, 'vec': np.array(self.vec), 'file_name': self.file_name}


class EsData(DataPoint):
    # 老版本的写入ES的数据字段，未启用
    question: str = ''
    answer: str = ''
    file_name: str = ''
    index: int
    chunk_type: str = Field(alias='type')
    url: str = ''
    parent_title: str = ''
    title: str = ''
    create_time: str = ''


class EsDataV2(DataPoint):
    # V2版本的写入ES字段，能确保字段数据类型符合预期
    question: str = ''
    answer: str = ''
    file_name: str = ''
    index: int = 0
    atom_index: int = 0
    agg_index: int = 0
    chunk_type: str = Field(alias='type')
    url: str = ''
    parent_title: str = ''
    title: str = ''
    create_time: str = ''
    doc_id: str


