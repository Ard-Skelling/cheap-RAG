# TODO: 单独剥离出Milvus存储
# 还没测过，不知道能不能用


import threading
import logging
from pymilvus import MilvusClient, DataType
from typing import List, Union, Optional
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class MilvusConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_prefix='milvus_')
    # Read env var "MILVUS_CONN_STR" automatically from env vars or .env file
    conn_str: str
    auto_release_seconds: float = 300.


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


# metaclass singleton used for singleton instance
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # we have not every built an instance before.  Build one now.
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            instance = cls._instances[cls]
            # here we are going to call the __init__ and maybe reinitialize.
            if hasattr(cls, '__allow_reinitialization') and cls.__allow_reinitialization:
                # if the class allows reinitialization, then do it
                instance.__init__(*args, **kwargs)  # call the init again
        return instance


class MilvusInterface(object):
    def __init__(self, config:MilvusConfig=None, collection:str=None) -> None:
        self.config = config or MilvusConfig()
        self.client = MilvusClient(self.config.conn_str)
        self.get_collection_map = dict()
        if collection:
            self.collection_map.update({collection: self.auto_release_collection(collection)})
        
    def auto_release_collection(self, collection_name):
        """
        闭包实现自动加载和超时释放集合
        :param collection_name: 集合名称
        :param release_timeout: 超时释放时间（秒），默认 300 秒
        :return: 一个函数，用于获取集合并启动超时释放
        """
        release_timeout = self.config.release_timeout
        self.client.load_collection(collection_name)
        timer = None

        def release():
            """释放集合"""
            nonlocal timer
            load_status = self.client.get_load_state(collection_name)
            if load_status.get('state', '') == '<LoadState: Loaded>':
                logging.info(f"Releasing collection '{collection_name}' due to timeout.")
                self.client.release_collection(collection_name)
                timer = None

        def get_collection():
            """获取集合并启动超时释放"""
            nonlocal timer
            load_status = self.client.get_load_state(collection_name)
            if load_status.get('state', '') == '<LoadState: NotLoad>':
                logging.info(f"Loading collection '{collection_name}'.")
                self.client.load_collection(collection_name)

            # 如果已有定时器，先取消
            if timer is not None:
                timer.cancel()

            # 启动新的定时器
            timer = threading.Timer(release_timeout, release)
            timer.start()

        return get_collection

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
        logging.info(f'Create collection successfully: {collection_name}')

    def insert(self, collection_name, data:List[dict], timeout=None, partition_name=''):
        
        res = self.client.insert(collection_name, data, timeout, partition_name)
        return res
    
    def upsert(self, collection_name, data:List[dict], timeout=None, partition_name=''):
        res = self.client.upsert(collection_name, data, timeout, partition_name)
        return res
    
    def delete(self, collection_name, ids=None, timeout=None, filter='', partition_name=''):
        res = self.client.delete(collection_name, ids, timeout, filter, partition_name)
        return res
        
    def search(
        self,
        collection_name: str,
        data: Union[List[list], list],
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[SearchParams] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        **kwargs,
    ) -> List[List[dict]]:
        search_params = search_params or SearchParams()
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
        res = self.client.query(collection_name, filter, output_fields, timeout, ids, partition_names, **kwargs)
        return res
    
    def describe_collection(self, collection_name):
        return self.client.describe_collection(collection_name)
    
    def describe_index(self, collection_name, index_name):
        return self.client.describe_index(collection_name, index_name)
    
    def drop_collection(self, collection_name):
        self.client.drop_collection(collection_name)

    def release_collection(self, collection_name=None):
        collection_name = collection_name or self.collection
        self.client.release_collection(collection_name)

    def close(self):
        self.client.close()


if __name__ == '__main__':
    ...