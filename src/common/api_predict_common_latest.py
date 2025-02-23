# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import logging
from flask import request

from configs import PublicEngineConfig
from common.deal_response import response_func, BadRequest

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import List, Union, Literal

__all__ = ["VecDataApiCommon"]


class BaseDealParams(object):
    """基础请求参数处理"""

    @staticmethod
    def get_json_param():
        return request.get_json()
    
    @staticmethod
    def data_validate_request(diff_request, data_dict:dict):
        csid = None
        try:
            data_dict = VecDataApiCommon.get_json_param()
            vd = diff_request(**data_dict)
            csid = vd.csid
        except Exception as e:
            # 校验必传参数
            logging.error(f"The required parameter csid is missing. ERROR: {e}")
            raise BadRequest(response_func(csid=csid, code="10002", message=f"The required parameter csid is missing. ERROR: {e}"))
        logging.info("Parse request parameter successfully.")
        return vd


# 展示collections参数验证类 url = f"http://{ip_port}/v1/model/VecDataEngine"
class VecBaseModel(BaseModel):
    model_config:ConfigDict = ConfigDict(extra='forbid')
    csid: str = Field(..., min_length=1, max_length=32, description="Client UUID")


# 创建collections参数验证类 url = f"http://{ip_port}/v1/model/VecDataCreateEngine"
class VecDataCreateEngineRequest(VecBaseModel):
    collections: List[str] = Field(..., min_items=1, description="List of collection names")
    version: int = Field(1, description='The retrival service version.')

# 文件名删除数据参数验证类 url = f"http://{ip_port}/v1/model/DataDeleteEngine"
class DataDeleteEngineRequest(VecBaseModel):
    domain: str = Field(default=f"{PublicEngineConfig.PUB_VEC_DB}", description="Name of the collection")
    file_name: Union[List[str], str] = Field(..., min_length=1, max_length=255, description="Name of the file")


# collections删除参数验证类 url = f"http://{ip_port}/v1/model/VecDataDeleteEngine"
class VecDataDeleteEngineRequest(VecBaseModel):
    collections: List[str] = Field(..., description="List of collection names")


# 领域检索参数验证类  url = f"http://{ip_port}/v1/model/FieldSearchEngine" done
class FieldSearchEngineRequest(VecBaseModel):
    domain: str = Field(..., description="Specify which database to query")
    output_fields: list = Field(["file_name"], description="User-specified fields to return, must be metadata fields, not vector data fields")

    @field_validator('output_fields')
    def validate_output_fields(cls, values):
        allowed_fields = ['id', "question", "answer", "file_name", "collection_name", 'index', 'type', 'url', 'parent_title', 'title', 'create_time']
        for v in values:
            if not isinstance(v, str):
                raise ValueError("Each item in output_fields must be a string")
            if v not in allowed_fields:
                raise ValueError(f'Field "{v}" is not allowed. Allowed fields are: {", ".join(allowed_fields)}')
        return values


# 文件名检索参数验证类 url = f"http://{ip_port}/v1/model/FileNameSearchEngine"
class FileNameSearchEngineRequest(VecBaseModel):
    domain: str = Field(f"{PublicEngineConfig.PUB_VEC_DB}", description="Specify which database to query")


# 写入数据参数验证类 url = f"http://{ip_port}/v1/model/DataInsertEngine" 
class DataInsertEngineRequest(VecBaseModel):
    document: str = Field(..., description="Base64 encoded document content")
    domain: str = Field(default=f"{PublicEngineConfig.PUB_VEC_DB}", description="collection for the data to insert")
    file_name: str = Field(..., min_length=1, max_length=255, description="Name of the file")
    file_meta: Union[dict, None] = Field(None,  description="file meta infomation")
    is_enhanced: bool = Field(False, description="Use LLM to generate question or not")
    version: int = Field(1, description='Retrival service version.')


# 精确检索数据验证类 url = f"http://{ip_port}/v1/model/DataPreciseSearchEngine" done
class DataPreciseSearchEngineRequest(VecBaseModel):
    question: Union[list, str] = Field(..., description="The question that needs to be queried in the document")
    search_field: Union[list, str] = Field(..., description="The field to search on (currently supports equality operations)")
    domain: str = Field(f"{PublicEngineConfig.PUB_VEC_DB}", description="Specify which database to query, defaults to the default database")
    topn: int = Field(0, ge=0, description="0: indicates returning all entities equal to search_field")
    output_fields: list = Field(["answer", "file_name", "index", "url"], description="Fields specified by the user to be returned, currently only supports ['answer','question', 'index', 'file_name']")
    
    @field_validator("output_fields", mode="after")
    def validate_output_fields(cls, values):
        allowed_fields = ["id", "question", "answer", "file_name", "collection_name", "index", "type", "url", "parent_title", "title", "create_time"]
        if not values:
            return allowed_fields # 默认不输出question values
        for v in values:
            if v not in allowed_fields:
                raise ValueError(f'Field "{v}" is not allowed. Allowed fields are: {", ".join(allowed_fields)}')
        return values
    
    @model_validator(mode='after')
    def match_question_search_field(cls, values):
        if isinstance(values.question, list) + isinstance(values.search_field, list) == 1:
            raise ValueError('question and search_field must be list together.')
        return values

# 检索数据参数验证类 url = f"http://{ip_port}/v1/model/DataSearchEngine" done
class DataSearchEngineRequest(VecBaseModel):
    question: str = Field(..., description="The question that needs to be queried.")
    domain: str = Field(f"{PublicEngineConfig.PUB_VEC_DB}", description="Specify the database to query, by default it queries the default database of 'pub_vec_db'.")
    threshold: float = Field(0.01, ge=0, le=1, description="Similarity matching threshold during retrieval, default is 0.8.")
    topn: int = Field(10, gt=0, description="Number of returned segments, default is 1.")
    output_fields: List[str] = Field(["answer", "file_name", "index", "url"], description="Fields specified by the user to be returned, currently only supports ['id','question', 'answer','file_name','collection_name', 'index', 'type', 'url', 'parent_title', 'title', 'create_time']")
    search_field: str = Field("vec", description="vector of the data, currently only supports ['vec', 'slice_vec', 'q_slice_vec']")
    use_rerank: bool = Field(False, description='Use reranker to rerank the semantic search result or not.')
    version: int = Field(1, description="Retrieval service version.")

    
    @field_validator("search_field", mode="before")
    def validate_search_field(cls, value):
        allowed_fields = ['vec', 'slice_vec', 'q_slice_vec']
        if value is None:
            value = "vec"  # Default value if None is provided
        elif value not in allowed_fields:
            raise ValueError(f'Field "{value}" is not allowed. Allowed fields are: {", ".join(allowed_fields)}')
        return value

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
        if values.method == 'put' and values.token != '428a56aeab878ce1eddb0c272784274b':
            raise ValueError(f'Token verification failed when putting to Minio.')
        if values.method == 'put' and not values.obj_base64:
            raise ValueError('obj_base64 is empty when putting to Minio.')
        return values


        
class VecDataApiCommon(BaseDealParams):
    """向量模块参数"""

    @staticmethod
    def parse_vecdata_request():
        """
            处理向量数据collection查询, 处理延时耗时处理接受请求
        """
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
    
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=csid, code="10001", message="Request method not support."))
        
        # 日志指纹, 必传用来定位日志文件
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(VecBaseModel, data_dict=data_dict)
        csid = vd.csid
            
        logging.info("Parse request parameter successfully.")
        return csid
    
    @staticmethod
    def parse_vecdata_create_request():
        """
            处理向量数据库创建, 处理延时耗时处理接受请求
        """
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
    
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=csid, code="10001", message="Request method not support."))
        
        # 日志指纹, 必传用来定位日志文件
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(VecDataCreateEngineRequest, data_dict=data_dict)
        csid = vd.csid
        collections = vd.collections
        version = vd.version
        logging.info("request parameter text: {}".format(collections))
        logging.info("Parse request parameter successfully.")
        return collections, csid, version
    
    @staticmethod
    def parse_vecdata_delete_request():
        """
            处理向量数据库删除, 处理延时耗时处理接受请求
        """
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
    
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=csid, code="10001", message="Request method not support."))
       
        # 日志指纹, 必传用来定位日志文件
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(VecDataDeleteEngineRequest, data_dict=data_dict)
        csid = vd.csid
        collections = vd.collections
         # 不能删除基础库
        if isinstance(collections, str):
            if collections == PublicEngineConfig.PUB_VEC_DB:
                raise BadRequest(response_func(csid=csid, code="10002", message="The database name cannot be the same as the default database."))
        elif isinstance(collections, list):
            if PublicEngineConfig.PUB_VEC_DB in collections:
                raise BadRequest(response_func(csid=csid, code="10002", message="The database name cannot be the same as the default database."))

        logging.info("request parameter text: {}".format(collections))
        logging.info("Parse request parameter successfully.")
        return collections, csid

    @staticmethod
    def parse_data_insert_request():
        """
            处理向量数据库插入数据, 处理延时耗时处理接受请求
        """
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
    
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=csid, code="10001", message="Request method not support."))
        
        # 日志指纹, 必传用来定位日志文件
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(DataInsertEngineRequest, data_dict=data_dict)
        csid = vd.csid
        document = vd.document
        domain = vd.domain
        file_name = vd.file_name
        file_meta = vd.file_meta
        is_enhanced = vd.is_enhanced
        version = vd.version
        logging.info("request parameter file_name: {}\n domain: {}".format(file_name, domain))
        logging.info("Parse request parameter successfully.")

        return document, domain, file_name, file_meta, csid, is_enhanced, version
    
    @staticmethod
    def parse_data_delete_request():
        """
            处理向量数据库删除数据, 处理延时耗时处理接受请求
        """
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=csid, code="10001", message="Request method not support."))
        
        # 日志指纹, 必传用来定位日志文件
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(DataDeleteEngineRequest, data_dict=data_dict)
        csid = vd.csid
        domain = vd.domain
        file_name = vd.file_name
        logging.info("request parameter domain: {}\nfile_name:{}".format(domain, file_name))
        logging.info("Parse request parameter successfully.")
        return domain, file_name, csid
    
    @staticmethod
    def parse_data_search_request():
        """数据查询接口"""
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=None, code="10001", message="Request method not support."))

        # question: 用户问题, topn: 返回的数量, csid: 定位日志的指纹
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(DataSearchEngineRequest, data_dict=data_dict)
        csid = vd.csid
        domain = vd.domain
        question = vd.question
        search_field = vd.search_field
        output_fields = vd.output_fields
        threshold = vd.threshold
        topn = vd.topn
        use_rerank = vd.use_rerank
        version = vd.version
        logging.info("request parameter question: {}\ndomain: {}\nsearch_field: {}\noutput_fields: {}\nthreshold: {}\ntopn: {}.".format(question, domain, search_field, output_fields, threshold, topn))
        logging.info("Parse request parameter successfully.")
        return question, domain, search_field, output_fields, threshold, topn, csid, use_rerank, version
    
    @staticmethod
    def parse_data_precise_search_request():
        """数据精确查询接口"""
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=None, code="10001", message="Request method not support."))

        # question: 用户问题, domain:库名, topn: 0精确搜索默认, csid: 定位日志的指纹
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(DataPreciseSearchEngineRequest, data_dict=data_dict)
        csid = vd.csid
        domain = vd.domain
        question = vd.question
        search_field = vd.search_field
        output_fields = vd.output_fields
        topn = vd.topn
        logging.info("request parameter question: {}\ndomain: {}\noutput_fields: {}\nsearch_field: {}\ntopn: {}.".format(question, domain, output_fields, search_field, topn))
        logging.info("Parse request parameter successfully.")
        return question, domain, output_fields, search_field, topn, csid 
    

    @staticmethod
    def parse_field_search_request():
        """数据精确查询接口"""
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=None, code="10001", message="Request method not support."))

        # domain:库名, csid: 定位日志的指纹
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(FieldSearchEngineRequest, data_dict=data_dict)
        csid = vd.csid
        domain = vd.domain
        output_fields = vd.output_fields
        logging.info("request parameter domain: {}\noutput_fields: {}.".format(domain, output_fields))
        logging.info("Parse request parameter successfully.")
        return domain, output_fields, csid 
    

    @staticmethod
    def parse_file_name_search_request():
        """数据精确查询接口"""
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=None, code="10001", message="Request method not support."))

        # domain:库名, csid: 定位日志的指纹
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(FileNameSearchEngineRequest, data_dict=data_dict)
        csid = vd.csid
        domain = vd.domain
        logging.info("request parameter domain: {}.".format(domain))
        logging.info("Parse request parameter successfully.")
        return domain, csid 
    
    @staticmethod
    def parse_obj_storage_request():
        """数据精确查询接口"""
        csid = None
        headers = request.headers
        # 打印头信息
        logging.info("request headers is: {}".format(str(headers)))
        if request.method != "POST":
            logging.error("Request method not support.")
            raise BadRequest(response_func(csid=None, code="10001", message="Request method not support."))

        # domain:库名, csid: 定位日志的指纹
        data_dict = VecDataApiCommon.get_json_param()
        vd = VecDataApiCommon.data_validate_request(ObjStorageEngineRequest, data_dict=data_dict)
        logging.info("Parse request parameter successfully.")
        return vd.bucket_name, vd.obj_path, vd.obj_base64, vd.csid, vd.method
    