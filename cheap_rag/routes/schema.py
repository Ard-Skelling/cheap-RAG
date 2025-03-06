from os import getenv
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import List, Union, Literal


# Local modules
from utils.logger import logger


# 展示collections参数验证类 url = f"http://{ip_port}/v1/model/VecDataEngine"
class VecBaseModel(BaseModel):
    model_config:ConfigDict = ConfigDict(extra='forbid')
    csid: str = Field(..., min_length=1, max_length=32, description="Client UUID")


# 创建collections参数验证类 url = f"http://{ip_port}/v1/model/VecDataCreateEngine"
class VecDataCreateEngineRequest(VecBaseModel):
    collections: Union[str, List[str]] = Field(..., min_items=1, description="List of collection names")
    version: int = Field(1, description='The retrival service version.')


# 文件名删除数据参数验证类 url = f"http://{ip_port}/v1/model/DataDeleteEngine"
class DataDeleteEngineRequest(VecBaseModel):
    domain: str = Field(..., description="Name of the collection")
    file_name: Union[List[str], str] = Field(..., min_length=1, max_length=255, description="Name of the file")


# collections删除参数验证类 url = f"http://{ip_port}/v1/model/VecDataDeleteEngine"
class VecDataDeleteEngineRequest(VecBaseModel):
    collections: Union[List[str], str] = Field(..., description="List of collection names")


# 领域检索参数验证类  url = f"http://{ip_port}/v1/model/FieldSearchEngine" done
class FieldSearchEngineRequest(VecBaseModel):
    domain: str = Field(..., description="Specify which database to query")
    output_fields: list = Field(["file_name"], description="User-specified fields to return, must be metadata fields, not vector data fields")

    @field_validator('output_fields')
    def validate_output_fields(cls, values):
        allowed_fields = ['text', "caption", "footnote", "file_name", "chunk_index", 'page_index', 'chunk_type', 'url', 'text_levle', 'create_time']
        for v in values:
            if not isinstance(v, str):
                raise ValueError("Each item in output_fields must be a string")
            if v not in allowed_fields:
                raise ValueError(f'Field "{v}" is not allowed. Allowed fields are: {", ".join(allowed_fields)}')
        return values


# 文件名检索参数验证类 url = f"http://{ip_port}/v1/model/FileNameSearchEngine"
class FileNameSearchEngineRequest(VecBaseModel):
    domain: str = Field(..., description="Specify which database to query")


# 写入数据参数验证类 url = f"http://{ip_port}/v1/model/DataInsertEngine" 
class DataInsertEngineRequest(VecBaseModel):
    document: str = Field(..., description="Base64 encoded document content")
    domain: str = Field(..., description="collection for the data to insert")
    file_name: str = Field(..., min_length=1, max_length=255, description="Name of the file")
    file_meta: Union[dict, None] = Field(None,  description="file meta infomation")


# 检索数据参数验证类 url = f"http://{ip_port}/v1/model/DataSearchEngine" done
class DataSearchEngineRequest(VecBaseModel):
    question: str = Field(..., description="The question that needs to be queried.")
    domain: str = Field(..., description="Specify the database to query, by default it queries the default database of 'pub_vec_db'.")
    threshold: float = Field(0.01, ge=0, le=1, description="Similarity matching threshold during retrieval, default is 0.8.")
    topn: int = Field(10, gt=0, description="Number of returned segments, default is 1.")
    output_fields: List[str] = Field(["text", "file_name", "chunk_index", "url"], description="Fields specified by the user to be returned.")


# 对象对储验证类 url = f"http://{ip_port}/v1/model/ObjStorageEngine"
class ObjStorageEngineRequest(VecBaseModel):
    token: str = ''
    bucket_name: str
    obj_path: str
    obj_base64: str = ''
    csid: str
    method: Literal['put', 'get'] = 'put'

    # 校验内部约定token，此接口不用户暴露，只对其它服务模块暴露
    # TODO: 改用更加健壮的加密方法
    @model_validator(mode='after')
    def check_token(cls, values):
        if values.method == 'put' and values.token != getenv('OBJ_TOKEN'):
            raise ValueError(f'Token verification failed when putting to Minio.')
        if values.method == 'put' and not values.obj_base64:
            raise ValueError('obj_base64 is empty when putting to Minio.')
        return values
