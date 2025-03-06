import sys
from pathlib import Path


BASE_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_PATH))


from typing import List, Union
from elasticsearch import Elasticsearch, helpers, NotFoundError
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


# Local modules
from configs.config_cls import ESConfig
from configs.config import ES_CONFIG
from utils.logger import logger


class Properties(BaseModel):
    timestamp: dict = Field({'type': 'date'}, alias='@timestamp')


class Mappings(BaseModel):
    properties: Properties


class Settings(BaseModel):
    ...


class ElasticSetting(BaseModel):
    suffix: str
    mappings: Mappings
    settings: Settings


class FrequentQuery:
    # TODO: General query schema patterns
    ...


class ElasticStorage:
    def __init__(self, config: ESConfig = None) -> None:
        self.config = config or ES_CONFIG
        self.__es = Elasticsearch(
            [f"http://{self.config.host}:{self.config.port}"],
            basic_auth=(self.config.user.get_secret_value(), self.config.pwd.get_secret_value()),
            verify_certs=False
        )


    # 创建索引
    def create_index(self, index_name:str, mappings:dict, settings:dict=None):
        """创建索引, 通过properties参数, 指定数据结构
        
        Params:
            index_name(str): 索引名, 一般与 milvus collection_name, minio bucket_name 等一致
            properties(dict): 字段属性, 形如:
                {
                    "timestamp": {"type": "date"},
                    "title": {"type": "text"},    # text型数据可以被分词和倒排索引, 支持BM25查询
                    "content": {"type": "text"},
                    "named_entity": {"type": "keyword"},    # 可传入字符串数组, 不会被分词, 适合不应当进行bm25查询的字段
                    "score": {"type": "float"},    # 可传入数字数组
                    "age": {"type": "integer"}
                }
        """
        # 定义索引映射
        mapping = {
            "mappings": mappings
        }
        if settings:
            mapping.update({"settings": settings})

        # 创建索引
        if not self.__es.indices.exists(index=index_name):
            self.__es.indices.create(index=index_name, body=mapping)
            logger.info(f"ES index {index_name} is created.")
        else:
            logger.info(f"ES index {index_name} has been created.")

    # 插入文档
    def insert_document(self, index_name:str, doc:dict, doc_id=None):
        kwargs = {
            'index': index_name,
            'body': doc
        }
        doc_id = doc_id or doc.pop('_id', None)
        if doc_id:
            kwargs.update({'id': doc_id})
        res = self.__es.index(**kwargs)
        logger.info(f"ES insert document with ID: {res['_id']}")

    # 批量插入文档
    def bulk_insert_documents(self, index_name:str, documents:List[dict]):
        actions = []
        for doc in documents:
            _id = doc.pop('_id', None)
            action = {
                "_index": index_name,
                "_source": doc
            }
            if _id:
                action.update({'_id': _id})
            actions.append(action)
        helpers.bulk(self.__es, actions)
        logger.info(f"ES bulk inserts {len(documents)} documents.")

    # 按id查询文档
    def get_doc_by_id(self, index_name:str, doc_ids:Union[List[Union[str, int]], Union[str, int]]):
        results = []
        if isinstance(doc_ids, list):
            res = self.__es.mget(index=index_name, body={'ids': doc_ids})
            for doc in res['docs']:
                results.append(doc['_source'])
        else:
            res = self.__es.get(index=index_name, id=doc_ids)
            results.append(res['_source'])
        # logger.info(f'ES文档: {results}')
        return results
    
    # 计数文档
    def count_documents(self, index_name:str, query=None):
        return self.__es.count(index=index_name, allow_no_indices=True, query=query)['count']

    
    @staticmethod
    def select_output_data(es_resp, output_fields:list=None):
        results = []
        for hit in es_resp["hits"]["hits"]:
            doc = hit['_source']
            output_data = dict()
            if output_fields:
                for k, v in doc.items():
                    if k in output_fields:
                        output_data[k] = v
                    if k == 'chunk_type' and 'type' in output_fields:
                        output_data['type'] = v
            else:
                output_data = doc
            results.append(output_data)
            # logger.info(f"ES文档 ID: {hit['_id']}, 内容: {output_data}")
        return results


    # 查询文档
    def search_documents(self, index_name:str, query:dict, output_fields:List[str]=None, size=10, **kwargs):
        size = 10000 if size == 0 else size
        body = {"query": query, "size": size}
        body.update(kwargs)
        res = self.__es.search(index=index_name, body=body)
        return self.select_output_data(res, output_fields)


    def multi_search(self, body:list, output_fields:list=None, index_name:str=None, is_flatten=True):
        """对同一索引同时发起多个查询。
        
        Params:
            body(list): 查询体，每两个元素表示一组查询。形如:
                [
                    {},  # 空行，表示使用 index_name 中指定的索引
                    {"query": {"term": {"field1": "value1"}}},
                    {},  # 空行，表示使用 index_name 中指定的索引
                    {"query": {"term": {"field2": "value2"}}},
                ]
            index_name(str): 索引名，如果为空，需要在body的奇数元素中指定，形如{"index": "my_index"}

        Returns:
            List[dict]: 查询结果返回的数据
        """
        res = self.__es.msearch(index=index_name, body=body)
        results = []
        for resp in res.body['responses']:
            result = self.select_output_data(resp, output_fields)
            if is_flatten:
                results.extend(result)
            else:
                results.append(result)
        return results


    def search_unique(self, index_name, field):
        query = {
            "size": 0,  # 不返回文档，只返回聚合结果
            "aggs": {
                "unique_values": {
                    "terms": {
                        "field": field,  
                        "size": 65536  
                    }
                }
            }
        }
        response = self.__es.search(index=index_name, body=query)
        unique_values = [bucket["key"] for bucket in response["aggregations"]["unique_values"]["buckets"]]
        return unique_values


    # 滚动查询文档，适用于大量数据的场景
    def scroll_search_documents(self, index_name:str, query:dict, output_fields:List[str]=None, size=100, scroll='1m'):
        size = 65536 if size == 0 else size
        body = {"query": query}
        # 使用滚动查询，在给定的时间窗口(scroll)内，轮询查出所有满足条件的结果
        res = self.__es.search(index=index_name, body=body, scroll=scroll, size=size)
        scroll_id = res['_scroll_id']
        results = []
        while True:
            hits = res['hits']['hits']
            if not hits:
                break
            for hit in hits:
                doc = hit['_source']
                output_data = dict()
                if output_fields:
                    for k, v in doc.items():
                        if k in output_fields:
                            output_data[k] = v
                else:
                    output_data = doc
                results.append(output_data)
                # logger.info(f"ES文档 ID: {hit['_id']}, 内容: {output_data}")
            res = self.__es.scroll(scroll_id=scroll_id)
        self.__es.clear_scroll(scroll_id=scroll_id)
        return results

    # 删除索引
    def delete_index(self, index_name):
        if self.__es.indices.exists(index=index_name):
            self.__es.indices.delete(index=index_name)
            logger.info(f"ES index {index_name} is deleted.")
        else:
            logger.info(f"ES index {index_name} dosen't exist.")

    # 删除文档
    def delete_doc(self, index_name, doc_id=None, query:dict=None):
        try:
            if doc_id:
                # res = self.get_doc_by_id(index_name, doc_id)
                # res = self.count_documents(index_name)
                if res:
                    res = self.__es.delete(index=index_name, id=doc_id)
                    logger.info(f"ES document {doc_id} is deleted.")
            else:
                res = self.search_documents(index_name, query=query, size=1)
                # res = self.count_documents(index_name)
                if res:
                    res = self.__es.delete_by_query(index=index_name, body={'query': query})
                    logger.info(f"ES deletes {res['deleted']} documents.")
        except NotFoundError as nf:
            logger.info("ES can't find target documents to delete.")

ES_STORAGE = ElasticStorage()


# 示例调用
if __name__ == "__main__":
    # import time


    # index_name = "test_index"
    # es = ElasticStorage()

    # # 创建索引
    # properties = {
    #     "question": {"type": "text"},
    #     "answer": {"type": "text"},
    #     "file_name": {"type": "keyword"},
    #     "index": {"type": "integer"},
    #     "chunk_type": {"type": "keyword"},
    #     "url": {"type": "keyword"},
    #     "parent_title": {"type": "text"},
    #     "title": {"type": "text"},
    #     "create_time": {"type": 'keyword'}
    # }
    # es.create_index(index_name, mappings={"properties": properties})

    # # 插入单个文档
    # doc = {
    #     "answer": "这是一个测试文本",
    #     "file_name": "sample.docx",
    #     "index": 89,
    #     "url": "images/page3_table0.jpg",

    # }
    # es.insert_document(index_name, doc=doc, doc_id='0a')

    # # 批量插入文档
    # documents = [
    #     {"answer": "文档 1", "file_name": "sampel_2.pdf", "index": 12, "_id": "0b"},
    #     {"answer": "这是文档 2 的内容", "file_name": "sample_3.pdf", "index": 2, "_id": "0c"}
    # ]
    # es.bulk_insert_documents(index_name, documents)

    # # 查询文档
    # query = {
    #     "match": {"answer": "内容"}
    # }
    # res = es.search_documents(index_name, query=query)
    # logger(res)

    # # 按id查询文档
    # ids = ['0b', '0c']
    # res = es.get_doc_by_id(index_name, ids)
    # logger(res)

    # # 使用多重查询
    # def build_query(i):
    #     query = {
    #         'bool': {
    #             'must': [
    #                 {'term': {'file_name': 'sample_3.pdf'}},
    #                 {'term': {'index': i}},
    #                 {'term': {'chunk_type': 'text'}}
    #             ]
    #         }
    #     }
    #     return {'query': query}
    
    # body = []
    # for i in [1, 2]:
    #     body.append(dict())
    #     body.append(build_query(i))

    # res = ES_STORAGE.multi_search(body, index_name=index_name, is_flatten=False)

    # # 删除文档
    # es.delete_doc(index_name, query={'terms': {'file_name': ['sample.pdf']}})


    # # 删除索引
    # es.delete_index(index_name)

    ...
