# -*- coding: utf-8 -*-
"""
    ***使用过程中的一些注意点***: 
        1. 在创建pymilvus的过程中, 主键字段对应的内容不能重复
        2. 在创建字段的时候, max_length不是字符串大小而是字符串的硬盘占用大小一般设置为最大字符✖3, https://github.com/milvus-io/pymilvus/issues/1667
        3. 查collection中实体的数量不要用 collection.num_entities(这个不准), 要精准的话需要 collection.query(expr="", output_fields=["count(*)"], consistency_level="Strong")
        4. utility.load_state(collection_name) == LoadState.NotLoad: 可以判断数据是饭否被加载到了内存或者显存上面
        5. 现在每次进来需要去查看collection种的总实体数量,这个操作非常耗时, 如果不查的话,如果collection为空的情况下查询会报错(准备每次新建库的时候插入一条空数据占位)
        6. 不要在创建集合的时候创建索引, 一定要在插入数据的时候再创建索引
"""

import time
import logging
import traceback
from collections import defaultdict
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
from pymilvus import (utility,
                      DataType, 
                      Collection, 
                      connections, 
                      FieldSchema, 
                      CollectionSchema)
from pymilvus.client.types import LoadState


# local module
from module.storage.elastic_storage import ES_STORAGE
from module.pipeline_v2.data_cls import MilvusDataV2


__all__=["PyMilvusInference"]


class FieldInputFetureEngiee(object):
    """不同collection字段设计"""
    def __init__(self):
        super(FieldInputFetureEngiee, self).__init__()
        # 19个领域库名称
        # 251343  19个库中的数据
    
    @staticmethod
    def design_domain_field(version=1):
        """不同领域知识字段设计"""
        class DomainFeture():
            def __init__(self, id, doc_id, question, answer, vec, slice_vec, q_slice_vec, file_name, index, type, collection_name, url, parent_title, title, create_time, version=1):
                self.main_field = vec  # 创建的索引字段
                self.id = id
                self.doc_id = doc_id
                self.question=question
                self.answer=answer
                self.vec=vec
                self.slice_vec = slice_vec
                self.q_slice_vec = q_slice_vec
                self.file_name = file_name
                self.index = index
                self.type = type
                self.collection_name = collection_name
                self.url = url
                self.parent_title = parent_title
                self.title = title
                # key list

                if version == 1:
                    self.field_list = [self.id, self.question, self.answer, self.vec, self.slice_vec, \
                                   self.q_slice_vec, self.file_name, self.index ,self.type, self.collection_name, \
                                    self.url, self.parent_title, self.title, create_time]
                if version == 2:
                    self.field_list = [self.id, self.doc_id, self.vec, self.file_name]


        id = FieldSchema(name="id", dtype= DataType.INT64, is_primary=True, auto_id=True, description="primary key")  # 主键
        question = FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1536, description="query text")  # query
        answer = FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=30000, description="answer text")  # query对应的答案或者切片
        vec = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=1024, description="vector of text")  # query对应的向量
        slice_vec = FieldSchema(name="slice_vec", dtype=DataType.FLOAT_VECTOR, dim=1024, description="vector of text")  # query对应的向量
        q_slice_vec = FieldSchema(name="q_slice_vec", dtype=DataType.FLOAT_VECTOR, dim=1024, description="vector of text")  # query对应的向量
        file_name = FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=300, description="file name text")  # 如果是qa则忽略, 如果是文章,则是其文件名
        index = FieldSchema(name="index", dtype=DataType.INT64, description="index")  # 如果是qa则忽略, 如果是文章,则存在在原文的切片索引
        type = FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=300, description="type text")  # qa 还是制度说明
        collection_name = FieldSchema(name="collection_name", dtype=DataType.VARCHAR, max_length=300, description="collection_name text")  # 数据领域标识
        url = FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=1000, description='the url of image or table') # 图片和表格的地址，后期换成url
        parent_title = FieldSchema(name='parent_title', dtype=DataType.VARCHAR, max_length=300, description='parent title') # 父级标题
        title = FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=300, description='title') # 当前片段的标题
        create_time = FieldSchema(name='create_time', dtype=DataType.VARCHAR, max_length=300, description='create time') # 当前片段的创建时间 
        doc_id = FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, description="atom chunk primary key")    # 原子化文本切片的id
        return DomainFeture(id, doc_id, question, answer, vec, slice_vec, q_slice_vec, file_name, index, type, collection_name, url, parent_title, title, create_time, version)


class PyMilvusInference(FieldInputFetureEngiee):
    """
        可以理解为 milvus是数据库, collections是数据表
    """
    def __init__(self, config):
        super(FieldInputFetureEngiee, self).__init__()
        
        # 定义参数
        self._define_params(config)
        # 连接milvus
        self._connet_milivus()

    def _define_params(self, config):
        self.ip = config.get("ip", "127.0.0.1")
        self.port = config.get("port", "19530")
        self.collections = config.get("collections", [])
       
        # 搜索时候向量计算的方式, 欧几里德距离: L2, 向量内积: IP, 余弦相似度: COSINE. ref: https://milvus.io/docs/metric.md
        self.metric_type = config.get("metric_type", "IP") 
        # 索引类型, ref: https://milvus.io/docs/index.md
        # self.index_type = config.get("index_type", "HNSW")
        self.index_type = config.get("index_type", "GPU_IVF_FLAT")
        # 簇大小, 要小于65535. ref: https://milvus.io/docs/index.md
        # https://milvus.io/docs/performance_faq.md  实验图比例选择 
        self.nlist = config.get("nlist", 128)
        assert self.nlist <= 65535, "nlist cannot exceed 65535"
        self.nprobe = config.get("nprobe", 8)
        assert self.nprobe <= self.nlist, "nprobe less than nlist"
        # 数据一致性级别
        self.consistency_level = config.get("consistency_level", "Strong")
    
    def _connet_milivus(self):
        """建立pymilvus的连接"""
        try:
            connections.connect(
                alias="default", 
                host="{}".format(self.ip),
                port="{}".format(self.port)
            )        
            logging.info("pymilvus connection successed.")
        except Exception as err:
            logging.info("Faild to pymilvus connection.")
            logging.error(traceback.format_exc())

    def _create_collection(self, collections, version=1):
        """创建collections"""
        try:
        
            for collection_name in collections:
                if not self._collection_is_exist(collection_name):
                    # 建立字段
                    Fetures = self.design_domain_field(version=version)
                    schema = CollectionSchema(fields=Fetures.field_list, auto_id=True, description=f"{collection_name} collection") 
                    Collection(name=collection_name, schema=schema, shards_num=2, enable_dynamic_field=True, consistency_level="Strong")  # 建表         
                    logging.info(f"create collect {collection_name} successful.")
            return "create success"
        except Exception as err:
            logging.error(traceback.format_exc())

    def _drop_collection(self, collection_name, schema=None):
        """删除collection"""
        try:
            # 先判断collection是否存在, 如果存在则删掉
            if self._collection_is_exist(collection_name):
                schema = self._get_collection_info(collection_name).schema
            logging.info("collection: {}, schema: {}".format(collection_name, schema))

            collection = Collection(
                name=collection_name,
                schema=schema 
            )
            # Drop the collection
            logging.info("origin collections: {}".format(self._get_collections))
            collection.drop()
            logging.info("update collections: {}".format(self._get_collections))
        except Exception as err:
            logging.error(traceback.format_exc())

    def _insert_data(self, data:List[MilvusDataV2], collection_name, field_index="vec", batch_size=1000, version=1):
        """
            插入数据
            param data: 需要插入的数据列表, 例如qa [[vec1, vec2], [query1, query2], [answer1, answer2]]
            param collection_name: 需要插入数据的collection
            param bacth_size: 每次插入的数据量大小, 防止一次插入过大失败
        """
        if version == 2:
            data = [d.to_ndarray() for d in data]
        try:
            collection = self._check_collection_state(collection_name)
            if not collection:
                logging.error("Faild to load collection.")
                return 
        
            if data and isinstance(data[0], dict):
                for i in range(0, len(data), batch_size):
                    sub_data = data[i:i+batch_size]
                    status = collection.insert(sub_data)
                    logging.info(f"insert info: {status}")
                    assert status.insert_count == status.succ_count

            elif data and isinstance(data[0], list):
                for i in range(0, len(data[0]), batch_size):
                    sub_data = []
                    for d in data:
                        sub_data.append(d[i:i+batch_size])
                    status = collection.insert(sub_data)
                    logging.info(f"insert info: {status}")
                    assert status.insert_count == status.succ_count
            else:
                raise NotImplementedError
            
            collection.flush()  # 固化到硬盘
            collection.load()  # 并将插入的load到内存

        except Exception as err:
            logging.error(f"from collection: {collection_name} insert data failed.")
            logging.error(traceback.format_exc())
            raise "insert error"
    
    def _delete_data(self, data, collection_name):
        """
            删除数据, 
            想根据什么字段删除就传该参数对应的值, 例如在qa问答中,
            qa有 id, vec, query, answer, 
            1. 如果想根据query删除则data如下格式
                data=[
                    {question: xxxx},
                ], 
            2. 如果批量删除则data如下格式
                data=[
                    {question: xxxx},
                    {question: xxxx}, 
                    ....
                ]
            3. 目前只考虑根据具体值进行删除, 如果后续有需求需要按照范围进行删除例如 10 < id < 100, 2020年8月 < date < 2023年10月则另外增加逻辑

            tips: 删除某一条数据, 只是删除掉主键对应的数据, 这一条主键id不会删除, 目前的主键id 2^63-1 所以是够用的

        """
        try:
            collection = self._check_collection_state(collection_name)
            if not collection:
                logging.error("Faild to load collection.")
                return 
        
            # 如果为空则不进行删除操作, 因为他还没有加载到内存上  
            entities_num = self._get_collection_entities(collection)
            logging.info(f"collection {collection_name} entities: {entities_num}.")   
            if not entities_num:
                logging.info(f"collection {collection_name} entities is 0.")      
                return
            
            for info in data:
                assert isinstance(info, dict), "input format error, params must is dict."
                # TODO: 以下删除方式可能导致超范围删除数据，建议改为and表达式，连接info字典中的过滤条件。
                for k, v in info.items():
                    delete_expr = f"{k} in [\'{v}\']"
                    status = collection.delete(expr=delete_expr)
                    logging.info(f"delete info: {status}")
            logging.info(f"from collection: {collection_name} delete data successful.")

            collection.flush()
            collection.load()
            utility.wait_for_loading_complete(collection_name=collection_name)

        except Exception as err:
            logging.error(f"from collection: {collection_name} delete data failed.")
            logging.error(traceback.format_exc())

    def _search_data(self, question, collection_name, search_field="vec", output_fields=["id", "question", "answer", "type", "index", "file_name", "collection_name", "url"], topn=10):
        """
            搜索相似向量结果
            注意这里是搜索相似向量和_query_data区别, _query_data是找数据库中已有的数据

            欧式距离: L2
            向量内积: IP
            余弦相似度: COSINE
            需要和创建的index保持一致
        """
        try:
            collection = self._check_collection_state(collection_name)
            
            if not collection:
                logging.error("Faild to load collection.")
                return 
          
            # 判断collection是否为空, 为空如果查询会报错, 故直接返回empty
            # 这个非常耗时, 因为全表扫描
            entities_num = self._get_collection_entities(collection)

            logging.info(f"collection {collection_name} entities: {entities_num}.")   
            if not entities_num:
                logging.info(f"collection {collection_name} entities is 0.")
                return []  

            search_params = {"metric_type": f"{self.metric_type}", "params": {"nprobe": f"{self.nprobe}"}}
            query_result = collection.search(
                data=question, 
                anns_field=search_field, 
                param=search_params, 
                limit=topn, 
                # demontrates the ways to reference a dynamic field.
                expr=None,
                # expr=f"file_name in ["南方电网自动化设备工程服务管理业务指导书.doc"]",
                # sets the names of the fields you want to retrieve from the search result.
                output_fields=output_fields, 
                consistency_level=f"{self.consistency_level}"
            )
            # todo L2, IP, COSNIN 的distance排序不一样, 后续需要根据不同的值进行排序处理, 目前以IP测试为主
            result = []
            for hits in query_result: 
                for hit in hits:
                    output_dict = {}
                    output_dict["score"] = round(hit.score, 4)
                    for field in output_fields:
                        output_dict[f"{field}"] = hit.entity.get(f"{field}")
                    result.append(output_dict)
            return result
        
        except Exception as err:
            logging.error(f"from collection: {collection_name} search data failed.")
            logging.error(traceback.format_exc())
            return []

    def _query_data(self, question, collection_name, operation="==", primary_key="question", output_fields=["id", "question", "answer", "type", "index", "file_name", "collection_name"], topn=0):
        """
            在库里找数据
            param question: 需要查找的question
            param collection_name: collection 库名
            param operation: 查找数据支持的操作符
            param primary_key: 查找的字段名
            param output_fields: 查找到数据需要返回字段对应的数据
            param topn: 查找返回的条数
        """
        try:
            collection = self._check_collection_state(collection_name)
            if not collection:
                logging.error("Faild to load collection.")
                return 
            
            entities_num = self._get_collection_entities(collection)
            logging.info(f"collection {collection_name} entities: {entities_num}.")  
            if not entities_num:
                logging.info(f"collection {collection_name} entities is 0.")
                return []  
            result = collection.query(
                expr=f"{primary_key} {operation} \'{question}\'" if primary_key != "id" else f"{primary_key} in [{question}]",
                # expr=f"{primary_key} in [{query}]" ,
                output_fields=output_fields,
                consistency_level="Strong",
                offset=0,
                limit=topn if topn else 16384
                )
            return result
        except Exception as err:
            logging.error(f"from collection: {collection_name} query data failed.")
            logging.error(traceback.format_exc())

    def _scalar_query_data(
        self,
        collection_name: str,
        expr: str,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        consistency_level=1,
        guarantee_timestamp: int = None,
        graceful_time: int = 5,
        start_offset: int = 0,
        end_limit: int = None,
        pool: ThreadPoolExecutor = None,
        batch_size: int = 16384
    ) -> List[dict]:
        try:
            collection = self._check_collection_state(collection_name)
            if not collection:
                logging.error("Faild to load collection.")
                return 
            if pool is None:
                result = collection.query(
                    expr=expr,
                    output_fields=output_fields,
                    timeout=timeout,
                    partition_names=partition_names,
                    consistency_level=consistency_level,
                    guarantee_timestamp=guarantee_timestamp,
                    graceful_time=graceful_time,
                    offset=start_offset,
                    limit=end_limit
                )
                return result
            else:
                if end_limit:
                    # 计算查询范围， 此时end_limit指最大返回条数
                    count = start_offset + end_limit
                else:
                    count = collection.query(expr='', output_fields=['count(*)'])
                    if not count:
                        logging.warning(f"Empty collection: {collection_name}")
                        return []
                    count = count[0]['count(*)']
                func = lambda offset: collection.query(
                    expr, 
                    output_fields, 
                    partition_names, 
                    timeout, 
                    consistency_level=consistency_level,
                    guarantee_timestamp=guarantee_timestamp,
                    graceful_time=graceful_time,
                    offset=offset,
                    limit=16384
                )
                final_data = []
                results = pool.map(func, [i for i in range(start_offset, count, batch_size)])
                for data in results:
                    final_data.extend(data)
                return final_data
        except Exception as err:
            logging.error(f"from collection: {collection_name} query data failed.")
            logging.error(traceback.format_exc())


    def _query_field_value(
        self,
        collection_name: str,
        output_fields: List[str],
        pool: ThreadPoolExecutor,
        batch_size:int = 16384
    ) -> Dict[str, List[str]]:
        """从milvus数据库中取出所有数据的对应字段去重后的值。
        注意：数据集分区中的数据不得大于16384条，否则无法取得全量数据
        """
        try:
            assert batch_size <= 16384, 'batch size must <= 16384'
            collection = self._check_collection_state(collection_name)
            if not collection:
                logging.error("Faild to load collection.")
                return 
            # 获取数据集数据条数
            count = collection.query(expr='', output_fields=['count(*)'])
            if not count:
                logging.warning(f"Empty collection: {collection_name}")
                return dict()
            count = count[0]['count(*)']
            func = lambda offset: collection.query('', output_fields, offset=offset, limit=batch_size)
            results = pool.map(func, [offset for offset in range(0, count, batch_size)])
            final_res = defaultdict(lambda: set())
            for data in results:
                for rec in data:
                    for field in output_fields:
                        final_res[field].add(rec[field])
            final_res = {k: list(v) for k, v in final_res.items()}
            return final_res

        except Exception as err:
            logging.error(f"from collection: {collection_name} query data failed.")
            logging.error(traceback.format_exc())

    def _delete_collection(self, collection_name):
        """删除collection"""
        try:
            if self._collection_is_exist(collection_name):
                utility.drop_collection(collection_name)
                logging.info(f"delete collection: {collection_name} successful.")                
        except Exception as err:
            logging.error("delete collection faild.")
            logging.error(traceback.format_exc())

    def _check_collection_state(self, collection_name):
        try:
            # 判断collection是否存在
            if not self._collection_is_exist(collection_name):
                logging.error("collection not exists.")
                return 
            
            collection = Collection(collection_name)  
            # 判断数据是否被加载到内存
            print("---------------------", utility.load_state(collection_name))
            if utility.load_state(collection_name) == LoadState.NotLoad: 
                # 检查index是否存在, 否则简历默认的index
                if not collection.has_index():
                    index = {
                        "index_type": f"{self.index_type}",
                        "metric_type": f"{self.metric_type}",
                        "params": {"nlist": f"{self.nlist}"},
                    }
                    # Which can either be a vector field or a scalar field
                    collection.create_index("vec", index)
                    # collection.create_index('slice_vec', index)
                    # collection.create_index('q_slice_vec', index)
                    logging.info("创建field vec index 成功")

                collection.flush()
                collection.load()
                utility.wait_for_loading_complete(collection_name=collection_name)
            return collection
        
        except Exception as err:
            logging.error(traceback.format_exc())
            return 

    def _create_index(self, collection, index_field):
        """创建index"""
        try:
            if not collection.has_index():
                index_params = {
                    "index_type": f"{self.index_type}",
                    "metric_type": f"{self.metric_type}",
                    "params": {"nlist": f"{self.nlist}"},
                }
                # Which can either be a vector field or a scalar field
                collection.create_index(field_name="vec", index_params=index_params)
                collection.create_index(field_name="slice_vec", index_params=index_params)
                collection.create_index(field_name="q_slice_vec", index_params=index_params)
        
                logging.info(f"创建field {index_field} index 成功")
            print("-------------collection-------------", collection.list_indexes())
            
        except Exception as err:
            logging.error(traceback.format_exc())

    @staticmethod
    def _get_collection_entities(collection):
        """精确获取collection中的数量"""
        entities = collection.query(expr="", output_fields=["count(*)"], consistency_level="Strong")
        return entities[0].get("count(*)", 0)

    @staticmethod
    def _get_collection_info(collection_name):

        """查看collection信息"""
        try:
            collection = Collection(collection_name)    
            logging.info(f"collection info is: {collection}")
            return collection
        except Exception as err:
            logging.error(traceback.format_exc())

    @staticmethod
    def _collection_is_exist(collection_name):
        """判断collection 是否存在"""
        if utility.has_collection(collection_name):
            return True
        return False

    @property
    def _get_collections(self):
        """获取当前连接中有多少个collection"""
        collection_list = utility.list_collections(timeout=None, using="default")
        return collection_list

if __name__=="__main__":
    ...



