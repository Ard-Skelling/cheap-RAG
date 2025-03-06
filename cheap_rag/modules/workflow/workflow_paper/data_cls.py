import numpy as np
from enum import Enum
from typing import Union, List, Literal
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, ConfigDict, field_validator, Field


class Data(BaseModel):
    model_config: ConfigDict = ConfigDict(extra='ignore', populate_by_name=True)


# Task data
class TaskData(Data):
    domain: str


class TaskMeta(TaskData):
    file_name: str
    csid: str = ''
    pdf_fp: Path = None
    json_fp: Path = None
    image_dir: Path = None


class Task(BaseModel):
    task_id: int
    task_meta: TaskMeta
    # 当前待执行的任务环节，对应Worker中的类方法名
    step: str
    status: Literal['pending', 'processing', 'completed', 'failed']
    result: dict = None


class FileData(TaskData):
    file_name: str = None
    file_base64: str = None


class OcrData(TaskData):
    file_name: str = None
    data: dict


# Intermediate data
class AtomChunk(Data):
    # MD5 hash of atomic chunk text
    # Generate_md5(f'{file_name}{chunk_id}{text}')
    atom_id: str
    text: str = ''    # Atomic text, question, table unique values, etc.
    agg_index: int    # Mapping to parent chunk's chunk_id
    raw_type: Literal[
        '',
        'table_body',
        'table_caption',
        'table_footnote',
        'image_caption',
        'image_footnote'
    ] = ''


class Chunk(Data):
    # MD5 hash text(text, equation) or image_path(table, image)
    # Generate_md5(f'{file_name}{text}{page_idx}') for text
    # Generate_md5(f'{file_name}{url}{page_idx}') for image_path
    chunk_id: str
    agg_index: int
    chunk_index: int = 0    # 0 for meta-info chunk, body chunk starts from 1
    page_index: int = 0
    chunk_type: str = 'text'


class AggChunk(Data):
    agg_id: str
    agg_index: int = 0
    page_index: int = 0
    text: str
    chunk_type: Literal[
        'text', 
        'title', 
        'summary', 
        'outline'
    ] = 'text'


class TextChunk(Chunk):
    text: str = ''
    text_level: int = 0
    chunk_type: str = 'text'


class EquationChunk(Chunk):
    text: str = ''
    text_format: str = 'latex'
    chunk_type: str = 'equation'


class ImageChunk(Chunk):
    url: str = ''
    caption: Union[list, str] = []
    footnote: Union[list, str] = []
    chunk_type: str = 'image'

    def list2str(self):
        fields = ['caption', 'footnote']
        for field in fields:
            v = getattr(self, field)
            if v:
                v = '\n'.join(v)
            else:
                v = ''
            setattr(self, field, v)


class TableChunk(ImageChunk):
    text: str = ''    # table body in html
    chunk_type: str = 'table'


# Final data
class MilvusData(Data):
    atom_id: str
    vec: List[Union[float, int]]
    agg_index: int
    file_name: str
    

class ESAtomData(Data):
    id: str = Field(alias='atom_id', serialization_alias='_id')
    text: str = ''    # Atomic text, question, table unique values, etc.
    agg_index: int    # Mapping to parent chunk's chunk_id
    file_name: str


class ESAggData(Data):
    id: str = Field(alias='agg_id', serialization_alias='_id')
    agg_index: int
    page_index: int = 0
    text: str
    file_name: str
    chunk_type: Literal[
        'text', 
        'title', 
        'summary', 
        'outline'
    ] = 'text'


class ESRawData(Data):
    id: str = Field(alias='chunk_id', serialization_alias='_id')
    agg_index: int
    chunk_index: int = 0
    page_index: int = 0
    text: str = ''
    caption: str = ''
    footnote: str = ''
    file_name: str
    chunk_type: Literal[
        'text', 
        'title', 
        'equation', 
        'table', 
        'image', 
        'summary', 
        'outline'
    ] = 'text'
    url: str = ''
    text_levle: int = 0
