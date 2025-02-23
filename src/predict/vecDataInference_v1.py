# -*- coding: utf-8 -*-
import os
import re
import time
import base64
import logging
import traceback
import numpy as np
from datetime import datetime
from predict import BaseInferenceEngine
from configs import PublicEngineConfig
from concurrent.futures import ThreadPoolExecutor
from module.parsers.predict import ParseFile
from module.vecdatas.predict import PyMilvusInference
from module.spliters.predict import SplitterInference
# from module.embeddings.predict import TextEmbeddingInference
from common import cost_time, response_func, BadRequest, send_request_server, send_request_embed_server


class VecDataInference(BaseInferenceEngine):
    """"""

    def __init__(self, config):
        self.config = config

        # pymilvus 配置
        self.vecdata_config = self.config.vecdata_config
        # 大模型配置
        self.llm_config = self.config.llm_config
        # 向量模型配置
        self.embed_config = self.config.embed_config

        self.thread_pool = ThreadPoolExecutor(max_workers=5)
        self.llm_thread_pool = ThreadPoolExecutor(max_workers=200)
        self.embed_thread_pool = ThreadPoolExecutor(max_workers=self.embed_config.get("thread_num", 16))   # 调用向量模型

        # 初始化文本切分器对象
        self._init_spliter_infer()
        # 初始化pymilvus向量数据库对象
        self._init_pymilvus_infer()
        # 初始化文本解析器对象
        self._init_parse_infer()

    def _init_spliter_infer(self):
        """初始化文本切分器对象"""        
        spliter_config = self.config.spliter_config
        self.spliter_infer = SplitterInference(spliter_config)
        logging.info("Init spliter infer successfully.")

    def _init_pymilvus_infer(self):
        """初始化向量数据库pymilvus所需要参数"""

        self.vecdata_infer = PyMilvusInference(self.vecdata_config)
        logging.info("Init vecdata infer successfully.")
    
    @cost_time
    def _init_parse_infer(self):
        """初始化文本解析器对象"""
        # parser_config = self.config.parser_config
        self.parser_infer = ParseFile()
        logging.info("Init parser infer successfully.")

    @cost_time
    def get_vecdata_result(self, csid):
        """获取向量数据库中collection的数量"""
        try:
            return self.vecdata_infer._get_collections
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="Failed to obtain the number of vector libraries."))

    @cost_time
    def create_vecdata(self, collections, csid):
        """创建collection"""
        try:
            return self.vecdata_infer._create_collection(collections)
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="create failed"))
    
    @cost_time
    def delete_vecdata(self, collections, csid):
        """删除collection"""
        try:
            for collection_name in collections:
                self.vecdata_infer._drop_collection(collection_name)
            return "delete success."
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="delete fialed"))
    
    @cost_time
    def construct_milvus_data_v1(self, result, domain, create_time, csid):
        """
            result是每一个切片对应的所有答案, 之所以不一次性送入到模型, 是因为当一个切片生成了很多问题的时候, 送入的字符总长度太长, 性能不好
        """
        try:
            answer_vec = None
            for i, res in enumerate(result):
                # todo, 看是想要用类似特殊标记还是想要直接把该条数据置为空向量

                # 思路1: 特殊标记占位                                                                
                question, answer = res["question"] if res.get("question", "") else "<unkown>", res["answer"] if res.get("answer", "") else "<unkown>"                                                               
                if i == 0:
                    answer_vec = cost_time(send_request_embed_server)(self.embed_config.get("url"), answer, csid)  # slice向量

                vec = cost_time(send_request_embed_server)(self.embed_config.get("url"), question, csid)  # question 向量
                q_slice_vec = cost_time(send_request_embed_server)(self.embed_config.get("url"), question + answer, csid)  # question + slice 向量
                
                res.update({
                    "vec": np.array(vec),
                    "slice_vec": np.array(answer_vec),
                    "q_slice_vec": np.array(q_slice_vec),
                    "collection_name": domain,
                    "create_time": create_time
                })

            return result

        except Exception as err:
            logging.error(traceback.format_exc())
            raise "Faild to construct data."
    
    @cost_time
    def construct_milvus_data(self, result, domain, create_time, csid):
        try:
            
            question_list = []
            for i, res in enumerate(result):
                # 特殊标记占位                                                                
                question, answer = res["question"] if res.get("question", "") else "<unkown>", res["answer"] if res.get("answer", "") else "<unkown>"                                                               
                if i == 0:
                    question_list.append(answer)
                    question_list.append(question)
                    question_list.append(question + answer)
                else:
                    question_list.append(question)
                    question_list.append(question + answer)

            # 调用向量模型生成构造的query. slice, query+slice 向量
            vecs = cost_time(send_request_embed_server)(self.embed_config.get("url"), question_list, csid)
            print("-----------vecs length--------------", len(vecs))
            answer_vec = np.array(vecs.pop(0))
            
            for i, res in zip(range(0, len(vecs), 2), result):
                vec_pairs = vecs[i: i+2]
                res.update({
                    "vec": np.array(vec_pairs[0]),
                    "slice_vec": answer_vec,
                    "q_slice_vec": np.array(vec_pairs[1]),
                    "collection_name": domain,
                    "create_time": create_time
                })

            return result

        except Exception as err:
            logging.error(traceback.format_exc())
            raise "Faild to construct data."

    @cost_time
    def process_insert_data(self, results, domain, csid, is_document=False):
        """
            多线程处理讲数据插入库中
            param results: 输出的数据条数
            param domain: 待插入的库名
        """
        data = []
        if not is_document:
            # step1: 首先判断库中是否存在该数据, 为了保持数据一致性,还是先删除再插入
            """
                删除再插入：
                优点：
                    简单直接：对于单个向量或者小批量向量的修改，删除旧向量后插入新向量是一种简单直观的方法。
                    数据一致性：如果修改涉及到向量数据和对应的元数据，删除再插入可以确保向量数据与元数据的一致性。
                缺点：
                    性能开销：删除操作需要查找向量，插入操作需要写入新数据，这都会带来额外的性能开销。
                    索引重建：如果向量集合有索引，删除向量后可能需要重新构建索引。
                直接修改：
                优点：
                    性能：对于大量数据的修改，直接修改可以减少读写操作，提高效率。
                    索引保持：如果修改不会影响索引结构，那么可以直接在现有索引上更新数据，无需重建索引。
                缺点：
                    复杂性：直接修改可能需要更复杂的逻辑来确保数据的一致性。
                    版本兼容性：Milvus 的不同版本对更新操作的支持和效率可能有所不同。
                    在决定使用哪种方式之前，你需要考虑以下因素：

                    数据量：如果修改的数据量很大，直接修改可能更高效。
                    修改频率：频繁的修改可能会导致性能问题，需要选择合适的方法。
                    索引结构：如果索引结构复杂，重建索引的成本较高，直接修改可能更合适。
            """
            for result in results:
                if isinstance(result, dict):
                    question, answer = result["question"], result["answer"]
                else:
                    question, answer = result[0], result[1]
                
                # 这里不要差直接删
                self.vecdata_infer._delete_data([{"question": question}], domain)
                
                if question:
                    vec = self.embed_infer.get_embedding(question)
                else:
                    vec = np.array([])
                data.append({
                    "question": question,
                    "answer": answer,
                    "vec": vec,
                    "file_name": "",
                    "index": 0,
                    "type": "qa",
                    "collection_name": domain,
                })

        else:
            create_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")  # 记录每一条的创建时间
            future_list = []
            temp_result = []
            for result in results:
                # 这里不要查直接删(耗时, 用时间标记)
                # self.vecdata_infer._delete_data([{"file_name": result["file_name"]}], domain)
                if not temp_result:
                    temp_result.append(result)
                else:
                    if result["answer"] == temp_result[-1]["answer"]:
                        temp_result.append(result)
                    else:
                        future = self.embed_thread_pool.submit(self.construct_milvus_data, temp_result, domain, create_time, csid)
                        future_list.append(future)
                        temp_result = []
            
            if temp_result:
                future = self.embed_thread_pool.submit(self.construct_milvus_data, temp_result, domain, create_time, csid)
                future_list.append(future)

            for future in future_list:
                result = future.result()
                data.extend(result)
        print("-------------data length----------------", len(data))
        self.vecdata_infer._insert_data(data, domain)     

    @cost_time
    def insert_data(self, document, domain, file_name, csid, batch=1000):
        """插入数据"""
        try:
            future_list = []
            bytes_data = base64.b64decode(document.encode("utf-8"))
            _, ext = os.path.splitext(file_name)
            file_type = ext.lower()
            if file_type in [".xlsx", ".xls", ".csv", ".json", ".txt"]:
                results = []
                if file_type in [".json"]:
                    for values in self.parser_infer.get_json_iters(bytes_data):
                        results.append({"question": values["question"], "answer": values["answer"]})
                        if results == batch:
                            future = self.thread_pool.submit(self.process_insert_data, results, domain, csid)
                            future_list.append(future)
                            results = []
                    
                elif file_type in [".xlsx", ".xls", ".csv"]:
                    for values in self.parser_infer.get_excel_iters(bytes_data, suffix=file_type):
                        results.append({"question": values[0], "answer": values[1]})
                        if results == batch:
                            future = self.thread_pool.submit(self.process_insert_data, results, domain, csid)
                            future_list.append(future)
                            results = []

                elif file_type in [".txt"]:
                    for values in self.parser_infer.get_txt_iters(bytes_data):
                        value_list = values.split("\\t")
                        results.append({"question": value_list[0], "answer": value_list[1]})
                        if results == batch:
                            future = self.thread_pool.submit(self.process_insert_data, results, domain, csid)
                            future_list.append(future)
                            results = []
                
                else:
                    raise NotImplementedError
                
                # 不够一个批次
                if results:
                    future = self.thread_pool.submit(self.process_insert_data, results, domain, csid)
                    future_list.append(future)
                    results = []
            else:
                
                start_time = time.time()
                # results = self.parser_infer.get_document_result(bytes_data, file_name, csid)

                import json
                with open("./ocrs/GB 14048.1-2012 低压开关设备和控制设备 第1部分 总则.pdf.json", "r", encoding="utf-8") as f:
                    results = json.load(f)

                print("------------results------------", len(results))
                print("ocr and llm extract question cost: {} s".format(time.time() - start_time))

                future = self.thread_pool.submit(self.process_insert_data, results, domain, csid, is_document=True)
                future_list.append(future)
           
            for future in future_list:
                _ = future.result()

            return "insert success"
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="insert failed"))
    
    @cost_time
    def delete_data(self, document, domain, file_name, csid):
        """删除数据"""
        try:
            bytes_data = base64.b64decode(document.encode("utf-8"))
            _, ext = os.path.splitext(file_name)
            file_type = ext.lower()
            if file_type in [".xlsx", ".xls", ".csv", ".json", ".txt"]:
                results = []
                if file_type in [".json"]:
                    for values in self.parser_infer.get_json_iters(bytes_data):
                        results.append({"question": values["question"], "answer": values["answer"]})
                elif file_type in [".xlsx", ".xls", ".csv"]:
                    for values in self.parser_infer.get_excel_iters(bytes_data, suffix=file_type):
                        results.append({"question": values[0], "answer": values[1]})
                elif file_type in [".txt"]:
                    for values in self.parser_infer.get_txt_iters(bytes_data):
                        value_list = values.split("\\t")
                        results.append({"question": value_list[0], "answer": value_list[1]})
                else:
                    raise NotImplementedError
                self.vecdata_infer._delete_data(results, domain)
            else:
                self.vecdata_infer._delete_data([{"file_name": file_name}], domain)

            return "delete success"
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="delete failed"))
    
    @cost_time
    def search_data(self, question, domain, threshold, topn, csid, search_field="vec", output_fields=[]):
        """查找数据"""
        try:
            final_result = []
            logging.info("collection_name: {}".format(domain))
            # 用户输入向量化
            vecs = cost_time(send_request_embed_server)(self.embed_config.get("url"), question, csid)
            query_vec = np.array(vecs[0])
            # 向量库检索
            search_data_list = cost_time(self.vecdata_infer._search_data)([query_vec], domain, search_field=search_field)

            # 根据得分进行排序
            search_data_list.sort(key=lambda x: x["score"])
            
            dup_answer = []
            dup_search_data_list = []
            for data in search_data_list[::-1]:
                if not dup_answer or data["answer"] not in dup_answer:
                    dup_answer.append(data["answer"])
                    dup_search_data_list.append(data)

            # for data in search_data_list[::-1][:topn]:
            for data in dup_search_data_list[:topn]:
                if data["score"] > threshold:
                    output_field_dict = {}
                    output_field_dict["score"] = data["score"]
                    for field in output_fields:
                        output_field_dict[field] = data.get(field, "")
                    final_result.append(output_field_dict)
            logging.info("final_result: {}".format(final_result))
            return final_result
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="search failed"))
    
    @cost_time
    def precise_search_data(self, question, domain, search_field, topn, csid, output_fields=[]):
        """根据指定字段精确查找数据"""
        try:
            final_result = []
            logging.info("collection_name: {}".format(domain))
            search_data_list = cost_time(self.vecdata_infer._query_data)(question, domain, primary_key=search_field, topn=topn)
            
            for data in search_data_list:
                output_field_dict = {}
                for field in output_fields:
                    output_field_dict[field] = data.get(field, "") 
                final_result.append(output_field_dict)
            logging.info("final_result: {}".format(final_result))
            return final_result
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="precise search failed"))
        
