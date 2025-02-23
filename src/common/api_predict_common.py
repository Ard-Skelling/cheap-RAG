# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import logging
from flask import request

from configs import PublicEngineConfig
from common.deal_response import response_func, BadRequest


__all__ = ["VecDataApiCommon"]


class BaseDealParams(object):
    """基础请求参数处理"""

    @staticmethod
    def getJsonParam(paramName, default=None):
        """
        处理表单请求参数
        :param paramName:
        :return: str | None
        """
        params = request.get_json()
        if params is None:
            params = request.form
        if params is not None:
            if paramName in params.keys():
                return params[paramName]
        return default

    @staticmethod
    def checkParamIsNull(paramList):
        """
        校验参数中是否有空值
        :param paramList: type: list、dict
        :return: bool
        """
        if isinstance(paramList, dict):
            for k, v in paramList.items():
                if v is None or v == "":
                    return True
        else:
            for param in paramList:
                if param is None or param == "":
                    return True
        return False

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
        csid = VecDataApiCommon.getJsonParam("csid")
        
        # 校验必传参数
        if not csid:
            logging.error("The required parameter csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required parameter csid is missing."))
        
        if not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=csid, code="10002", message="Wrong data type."))
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
        collections = VecDataApiCommon.getJsonParam("collections")
        csid = VecDataApiCommon.getJsonParam("csid")
    
        # 校验必传参数
        if not csid:
            logging.error("The required parameter csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required parameter csid is missing."))
        
        if not isinstance(collections, list) or \
            not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=csid, code="10002", message="Wrong data type."))
        
        logging.info("request parameter text: {}".format(collections))
        logging.info("Parse request parameter successfully.")
        return collections, csid
    
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
        collections = VecDataApiCommon.getJsonParam("collections")
        csid = VecDataApiCommon.getJsonParam("csid")
        
        # 校验必传参数
        if not csid:
            logging.error("The required parameter csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required parameter csid is missing."))
        
        if not isinstance(collections, (list, str)) or \
            not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=csid, code="10002", message="Wrong data type."))
        
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
        document = VecDataApiCommon.getJsonParam("document")
        domain = VecDataApiCommon.getJsonParam("domain") if VecDataApiCommon.getJsonParam("domain") else "pub_vec_db"
        file_name = VecDataApiCommon.getJsonParam("file_name")
        file_meta = VecDataApiCommon.getJsonParam("file_meta")    # dict, 文件级的元数据
        csid = VecDataApiCommon.getJsonParam("csid")
        
        # 校验必传参数
        if not csid:
            logging.error("The required parameter csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required parameter csid is missing."))
        
        if not isinstance(document, str) or \
            not isinstance(domain, str) or \
            not isinstance(file_name, str) or \
            not (file_meta is None or isinstance(file_meta, dict)) or \
            not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=csid, code="10002", message="Wrong data type."))
        
        logging.info("request parameter file_name: {}\n domain: {}".format(file_name, domain))
        logging.info("Parse request parameter successfully.")
        return document, domain, file_name, file_meta, csid
    
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
        domain = VecDataApiCommon.getJsonParam("domain") if VecDataApiCommon.getJsonParam("domain") else "pub_vec_db"
        file_name = VecDataApiCommon.getJsonParam("file_name")
        csid = VecDataApiCommon.getJsonParam("csid")
        
        # 校验必传参数
        if not csid:
            logging.error("The required parameter csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required parameter csid is missing."))
        
        if not isinstance(domain, str) or \
            not (isinstance(file_name, str) or isinstance(file_name, list)) or \
            not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=csid, code="10002", message="Wrong data type."))
        
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
        question = VecDataApiCommon.getJsonParam("question")
        domain = VecDataApiCommon.getJsonParam("domain") if VecDataApiCommon.getJsonParam("domain") else PublicEngineConfig.PUB_VEC_DB
        search_fields = VecDataApiCommon.getJsonParam("search_fields", "vec")
        output_fields = VecDataApiCommon.getJsonParam("output_fields", ["answer", "file_name", "index"])
        threshold = VecDataApiCommon.getJsonParam("threshold", 0.8)
        topn = VecDataApiCommon.getJsonParam("topn", default=1)
        csid = VecDataApiCommon.getJsonParam("csid")

        if not all([question, csid]):
            logging.error("The required parameter question or csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required parameter question or csid is missing."))
        
        if not isinstance(question, str) or \
            not isinstance(domain, str) or \
            not isinstance(output_fields, list) or \
            not isinstance(threshold, float) or \
            not isinstance(topn, int) or \
            not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=csid, code="10002", message="Wrong data type."))

        logging.info("request parameter question: {}\ndomain: {}\nsearch_fields: {}\noutput_fields: {}\nthreshold: {}\ntopn: {}.".format(question, domain, search_fields, output_fields, threshold, topn))
        logging.info("Parse request parameter successfully.")
        return question, domain, search_fields, output_fields, threshold, topn, csid 
    
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

        # question: 用户问题, domain:库名, topn: 返回的数量, csid: 定位日志的指纹
        question = VecDataApiCommon.getJsonParam("question")
        domain = VecDataApiCommon.getJsonParam("domain") if VecDataApiCommon.getJsonParam("domain") else PublicEngineConfig.PUB_VEC_DB
        output_fields = VecDataApiCommon.getJsonParam("output_fields", ["answer", "file_name", "index"])
        search_field = VecDataApiCommon.getJsonParam("search_field")
        topn = VecDataApiCommon.getJsonParam("topn", default=0)
        csid = VecDataApiCommon.getJsonParam("csid")

        if not all([question, csid]):
            logging.error("The required parameter question or csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required parameter question or csid is missing."))
        
        if not isinstance(question, str) or \
            not isinstance(domain, str) or \
            not isinstance(output_fields, list) or \
            not isinstance(search_field, str) or \
            not isinstance(topn, int) or \
            not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=None, code="10002", message="Wrong data type."))

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

        # question: 用户问题, domain:库名, topn: 返回的数量, csid: 定位日志的指纹
        domain = VecDataApiCommon.getJsonParam("domain") if VecDataApiCommon.getJsonParam("domain") else PublicEngineConfig.PUB_VEC_DB
        output_fields = VecDataApiCommon.getJsonParam("output_fields", ["file_name"])
        csid = VecDataApiCommon.getJsonParam("csid")

        if not csid:
            logging.error("The required parameter csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required paramete csid is missing."))
        
        if not isinstance(domain, str) or \
            not isinstance(output_fields, list) or \
            not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=None, code="10002", message="Wrong data type."))

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

        # question: 用户问题, domain:库名, topn: 返回的数量, csid: 定位日志的指纹
        domain = VecDataApiCommon.getJsonParam("domain") if VecDataApiCommon.getJsonParam("domain") else PublicEngineConfig.PUB_VEC_DB
        csid = VecDataApiCommon.getJsonParam("csid")

        if not csid:
            logging.error("The required parameter csid is missing.")
            raise BadRequest(response_func(csid=csid, code="10002", message="The required paramete csid is missing."))
        
        if not isinstance(domain, str) or \
            not isinstance(csid, str):
            logging.error("Wrong data type.")
            raise BadRequest(response_func(csid=None, code="10002", message="Wrong data type."))

        logging.info("request parameter domain: {}.".format(domain))
        logging.info("Parse request parameter successfully.")
        return domain, csid 