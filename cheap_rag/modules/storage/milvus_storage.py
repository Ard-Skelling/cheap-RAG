# TODO: 单独剥离出Milvus存储
# 还没测过，不知道能不能用


import threading
from pymilvus import MilvusClient, DataType
from typing import List, Union, Optional
from pydantic import BaseModel, ConfigDict


# local modules
from configs.config_cls import MilvusConfig
from configs.config import MILVUS_CONFIG
from utils.logger import logger


class Field(BaseModel):
    model_config = ConfigDict(extra='allow')

    field_name: str
    datatype: DataType


class IDField(Field):
    # Use md5 of text content in default
    datatype: DataType = DataType.VARCHAR
    is_primary: bool = True
    max_length: int = 32


class StrField(Field):
    datatype: DataType = DataType.VARCHAR
    max_length: int = 32768


class VectorField(Field):
    datatype: DataType = DataType.FLOAT_VECTOR
    dim: int


class SparseVectorField(Field):
    datatype: DataType = DataType.SPARSE_FLOAT_VECTOR


class IndexSetting(BaseModel):
    model_config = ConfigDict(extra='allow')

    field_name: str


class CpuVectorIndexSetting(IndexSetting):
    """CPU version for dense vector search with HNSW algorithm"""
    metric_type: str = 'IP'
    index_name: str = 'cpu_dense_vec_index'
    index_type: str = 'HNSW'
    params: dict = {
        'M': 16,
        'efConstruction': 200
    }  


class GpuVectorIndexSetting(IndexSetting):
    """GPU accelerator for dense vector search with CAGRA algorithm"""
    metric_type: str = 'IP'
    index_name: str = 'gpu_dense_vec_index'
    index_type: str = 'GPU_CAGRA'
    params: dict = {
        'intermediate_graph_degree': 64,
        'graph_degree': 32
    }

class SparseVectorIndexSetting(IndexSetting):
    """Used for sparse vector search, compatible with algorithms such as BM25, SPLADE, etc."""
    metric_type: str = 'IP'
    index_name: str = 'sparse_vector_index'
    index_type: str = 'SPARSE_INVERTED_INDEX'
    params: dict = {"drop_ratio_build": 0.2}


class SearchParams(BaseModel):
    metric_type:str = 'IP'
    params: dict = dict()


class CpuHnswSearchParams(SearchParams):
    params: dict = {'ef': 100}


class GpuCagraSearchParams(SearchParams):
    params: dict = {
        "itopk_size": 128,
        "search_width": 4,
        "min_iterations": 0,
        "max_iterations": 0,
        "team_size": 0
    }


class SparseSearchParams(BaseModel):
    metric_type:str = 'IP'
    params: dict = {"drop_ratio_search": 0.2}


class MilvusInterface(object):
    def __init__(self, config:MilvusConfig=None) -> None:
        self.config = config or MILVUS_CONFIG
        self.client = MilvusClient(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            token=self.config.token
        )
        self.collection_map = dict()
        collection = self.config.default_collection
        if collection:
            self.collection_map.update({collection: self.auto_release_collection(collection)})
        

    def auto_release_collection(self, collection_name):
        """
        闭包实现自动加载和超时释放集合
        :param collection_name: 集合名称
        :return: get_collection, manual_release函数，用于获取集合并启动超时释放
        """
        release_timeout = self.config.auto_release_seconds
        self.client.load_collection(collection_name)
        timer = None

        def release():
            """释放集合"""
            nonlocal timer
            load_status = self.client.get_load_state(collection_name)
            if load_status.get('state', '') == '<LoadState: Loaded>':
                logger.info(f"Releasing collection '{collection_name}' due to timeout.")
                self.client.release_collection(collection_name)
                timer = None
                self.collection_map.pop(collection_name, None)

        def get_collection():
            """获取集合并启动超时释放"""
            nonlocal timer
            load_status = self.client.get_load_state(collection_name)
            if load_status.get('state', '') == '<LoadState: NotLoad>':
                logger.info(f"Loading collection '{collection_name}'.")
                self.client.load_collection(collection_name)

            # 如果已有定时器，先取消
            if timer is not None:
                timer.cancel()

            # 启动新的定时器
            timer = threading.Timer(release_timeout, release)
            timer.start()

        def manual_release():
            """手动触发释放集合并停止定时器"""
            nonlocal timer
            if timer is not None:
                timer.cancel()  # 取消定时器
            release()  # 调用释放集合的逻辑

        return get_collection, manual_release



    def create_default_collection(self, collection_name:str, dimension:int):
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension
        )


    def create_collection(
        self,
        collection_name: str,
        fields: List[Field],
        index_settings: List[IndexSetting],
        auto_id: bool = False,
        enable_dynamic_field: bool = True
    ):
        # Create schema
        schema = MilvusClient.create_schema(
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic_field,
        )
        # Add fields to schema
        for field in fields:
            schema.add_field(**field.model_dump())
        # Prepare index parameters
        index_params = self.client.prepare_index_params()
        # Add indexes
        for index_setting in index_settings:
            index_params.add_index(**index_setting.model_dump())
        # Create a collection
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f'Create collection successfully: {collection_name}')


    def load_collection(self, collection_name):
        if collection_name in self.collection_map:
            # execute get_collection method, whitch is the first element
            self.collection_map[collection_name][0]()
        else:
            self.collection_map[collection_name] = self.auto_release_collection(collection_name)


    def insert(self, collection_name, data:List[dict], timeout=None, partition_name=None):
        self.load_collection(collection_name)
        res = self.client.insert(collection_name, data, timeout, partition_name)
        return res
    

    def upsert(self, collection_name, data:List[dict], timeout=None, partition_name=None):
        self.load_collection(collection_name)
        res = self.client.upsert(collection_name, data, timeout, partition_name)
        return res
    

    def delete(self, collection_name, ids=None, timeout=None, filter='', partition_name=None):
        res = self.client.delete(collection_name, ids, timeout, filter, partition_name)
        return res
        

    def search(
        self,
        collection_name: str,
        data: List[List[Union[int, float]]],
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[SearchParams] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        **kwargs,
    ) -> List[List[dict]]:
        self.load_collection(collection_name)
        search_params = search_params or SearchParams()
        if not isinstance(data[0], list):
            data = [data]
        res = self.client.search(collection_name, data, filter, limit, \
            output_fields, search_params.model_dump(), timeout, partition_names, anns_field, **kwargs)
        return res
    

    def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        ids: Optional[Union[List, str, int]] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[dict]:
        self.load_collection(collection_name)
        res = self.client.query(collection_name, filter, output_fields, timeout, ids, partition_names, **kwargs)
        return res
    

    def describe_collection(self, collection_name):
        return self.client.describe_collection(collection_name)
    

    def describe_index(self, collection_name, index_name):
        return self.client.describe_index(collection_name, index_name)
    

    def drop_collection(self, collection_name):
        self.client.drop_collection(collection_name)


    def release_collection(self, collection_name):
        # execute manual_release method, which is the second element
        self.collection_map[collection_name][1]()
        

    def close(self):
        self.client.close()



MILVUS_STORAGE = MilvusInterface()


if __name__ == '__main__':
    # coll = 'test'

    # # create collection
    # fields = [
    #     IDField(
    #         field_name='doc_id'
    #     ),
    #     StrField(
    #         field_name='color',
    #         max_length=512
    #     ),
    #     VectorField(
    #         field_name='vec',
    #         dim=5
    #     )
    # ]
    # index_settings = [
    #     CpuVectorIndexSetting(
    #         field_name='vec'
    #     )
    # ]

    # res = MILVUS_STORAGE.create_collection(
    #     collection_name=coll,
    #     fields=fields,
    #     index_settings=index_settings,
    #     enable_dynamic_field=False
    # )


    # # insert data
    # data=[
    #     {"doc_id": '0', "vec": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "pink_8682"},
    #     {"doc_id": '1', "vec": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "color": "red_7025"},
    #     {"doc_id": '2', "vec": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], "color": "orange_6781"},
    #     {"doc_id": '3', "vec": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345], "color": "pink_9298"},
    #     {"doc_id": '4', "vec": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106], "color": "red_4794"},
    #     {"doc_id": '5', "vec": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955], "color": "yellow_4222"},
    #     {"doc_id": '6', "vec": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987], "color": "red_9392"},
    #     {"doc_id": '7', "vec": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052], "color": "grey_8510"},
    #     {"doc_id": '8', "vec": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336], "color": "white_9381"},
    #     {"doc_id": '9', "vec": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608], "color": "purple_4976"}
    # ]
    # MILVUS_STORAGE.insert(coll, data)

    # # query data
    # _filter = 'color in ["pink_9298", "grey_8510"]'
    # res = MILVUS_STORAGE.query(coll, filter=_filter)

    # # search data
    # query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]
    # res = MILVUS_STORAGE.search(coll, query_vector, limit=5)

    # # delete data
    # res = MILVUS_STORAGE.delete(coll, ids=['0', '1'])

    # MILVUS_STORAGE.release_collection(coll)

    # # delete collection
    # res = MILVUS_STORAGE.drop_collection(coll)


    # MILVUS_STORAGE.close()

    ...