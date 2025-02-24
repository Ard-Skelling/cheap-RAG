import sys
from pathlib import Path


BASE_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_PATH))


from typing import List, Union
from elasticsearch import Elasticsearch, helpers, NotFoundError
from datetime import datetime
from configs.vector_database_config import GLOBAL_CONFIG


ES_CONFIG = GLOBAL_CONFIG.es_config


class FrequentQuery:
    # TODO: 收录一些常用的查询范式
    ...


class ElasticStorage:
    def __init__(self) -> None:
        self.__es = Elasticsearch(
            [f"http://{ES_CONFIG['host']}:{ES_CONFIG['port']}"],
            basic_auth=(ES_CONFIG['user'], ES_CONFIG['pwd']),
            verify_certs=False
        )


    # 创建索引
    def create_index(self, index_name:str, mappings:dict=None, settings:dict=None, version=1):
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
        mappings = mappings or {
            "properties": {
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "file_name": {"type": "keyword"},
                "index": {"type": "integer"},
                "atom_index": {"type": "integer"},
                "agg_index": {"type": "integer"},
                # chunk_type 支持 text, table, image, qa, outline, summary, atom_text, agg_text几种形式
                # atom_text 是原子化的文本片，是向量检索的最小单元
                # agg_text 是长达20000字左右的大文本片，是真正返回的片段
                "chunk_type": {"type": "keyword"},
                "url": {"type": "keyword"},
                "parent_title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "create_time": {"type": 'keyword'}
            }
        }
        if version == 2:
            mappings['properties'].update({
                "doc_id": {"type": "keyword"}
            })
        mapping = {
            "mappings": mappings
        }
        settings = settings or {
            "analysis": {
                "tokenizer": {
                    "ik_max_word": {
                        "type": "ik_max_word"  # 使用 ik_max_word 分词器
                    }
                },
                "analyzer": {
                    "default": {
                        "type": "custom",
                        "tokenizer": "ik_max_word"
                    }
                }
            }
        }
        mapping.update({"settings": settings})

        # 创建索引
        if not self.__es.indices.exists(index=index_name):
            self.__es.indices.create(index=index_name, body=mapping)
            print(f"ES索引 {index_name} 已创建")
        else:
            print(f"ES索引 {index_name} 已存在")

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
        print(f"ES插入文档 ID: {res['_id']}")

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
        print(f"ES批量插入了 {len(documents)} 条文档")

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
        # print(f'ES文档: {results}')
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
            # print(f"ES文档 ID: {hit['_id']}, 内容: {output_data}")
        return results


    # 查询文档
    def search_documents(self, index_name:str, query:dict, output_fields:List[str]=None, size=10, **kwargs):
        body = {"query": query}
        body.update(kwargs)
        size = 10000 if size == 0 else size
        res = self.__es.search(index=index_name, body=body, size=size)
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
                # print(f"ES文档 ID: {hit['_id']}, 内容: {output_data}")
            res = self.__es.scroll(scroll_id=scroll_id)
        self.__es.clear_scroll(scroll_id=scroll_id)
        return results

    # 删除索引
    def delete_index(self, index_name):
        if self.__es.indices.exists(index=index_name):
            self.__es.indices.delete(index=index_name)
            print(f"ES索引 {index_name} 已删除")
        else:
            print(f"ES索引 {index_name} 不存在")

    # 删除文档
    def delete_doc(self, index_name, doc_id=None, query:dict=None):
        try:
            if doc_id:
                # res = self.get_doc_by_id(index_name, doc_id)
                # res = self.count_documents(index_name)
                if res:
                    res = self.__es.delete(index=index_name, id=doc_id)
                    print(f"ES文档 {doc_id} 已删除")
            else:
                res = self.search_documents(index_name, query=query, size=1)
                # res = self.count_documents(index_name)
                if res:
                    res = self.__es.delete_by_query(index=index_name, body={'query': query})
                    print(f"ES删除了 {res['deleted']} 条文档")
        except NotFoundError as nf:
            print('ES未找到待删除文档')

ES_STORAGE = ElasticStorage()


# 示例调用
if __name__ == "__main__":
    # import time


    # index_name = "little_q_241224"
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
    #     "match": {"answer": "制度考核"}
    # }
    # res = es.search_documents(index_name, query=query)
    # print(res)

    # # 按id查询文档
    # ids = ['0b', '0c']
    # res = es.get_doc_by_id(index_name, ids)
    # print(res)

    # # 使用多重查询
    # def build_query(i):
    #     query = {
    #         'bool': {
    #             'must': [
    #                 {'term': {'file_name': 'GJB-2809A-2005 部队油库加油站设计与施工规范.pdf'}},
    #                 {'term': {'index': i}},
    #                 {'term': {'chunk_type': 'text'}}
    #             ]
    #         }
    #     }
    #     return {'query': query}
    
    # body = []
    # for i in [-1, 10000]:
    #     body.append(dict())
    #     body.append(build_query(i))

    # res = ES_STORAGE.multi_search(body, index_name=index_name, is_flatten=False)

    # # 删除文档
    # es.delete_doc(index_name, query={'terms': {'file_name': ['negtive_sample.pdf']}})


    # # 删除索引
    # es.delete_index(index_name)

    ...
