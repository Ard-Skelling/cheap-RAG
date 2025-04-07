import sys
import os
import json
from io import BytesIO
from datetime import timedelta
from minio import Minio
from minio.deleteobjects import DeleteObject


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Local modules
from utils.logger import logger
from configs.config_cls import MinioConfig
from configs.config import MINIO_CONFIG
from concurrent.futures import ThreadPoolExecutor


class MinioStorage:
    """Minio 对象存储。适合用于分布式存储文件数据，例如图片、文档、视频、音频、模型权重、本地安装包等等"""
    def __init__(self, config: MinioConfig = None) -> None:
        self.config = config or MINIO_CONFIG
        self.client = Minio(
            endpoint=f"{self.config.host}:{self.config.port}",
            access_key=self.config.ak.get_secret_value(),
            secret_key=self.config.sk.get_secret_value(),
            secure=False 
        )
        bucket_name = self.config.bucket
        bucket_name_ocr = self.config.bucket_ocr
        self.create_bucket(bucket_name)
        self.create_bucket(bucket_name_ocr)
        self.minio_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

    def create_bucket(self, bucket_name):
        is_existed = self.client.bucket_exists(bucket_name)
        if not is_existed:
            self.client.make_bucket(bucket_name)
            logger.info(f'create minio bucket {bucket_name} seccessfully.')

    def put_object(
        self,
        object_name:str,
        data:bytes,
        bucket_name:str=None,
        metadata:dict=None,
        length=-1,
        part_size=10*1024*1024,
        content_type='application/octet-stream',
        **kwargs
    ):
        """用来将一个二进制对象存储到Minio中。
        Params:
            object_name(str): 对象存储的名称，形如: test_collection/test_doc.doc
            data(bytes): 二进制文件内容
            metadata(dict): 文件的元数据，可以是一个自定义的JSON字典，用于描述文件的相关信息
            length(int): 二进制文件的长度
            part_size(int): 分部上传时部分的大小
            content_type(str): 指定的文件格式
        """
        bucket_name = bucket_name or self.config.bucket
        if isinstance(data, dict):
            data = json.dumps(data, indent=4, ensure_ascii=False)
            data = BytesIO(data.encode('utf-8'))
        result = self.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=data, 
            metadata=metadata,
            length=length,
            part_size=part_size,
            content_type=content_type,
            **kwargs
        )
        logger.debug(f'Uploaded to Minio: {object_name}')

    def get_object(
            self,
            object_name:str,
            bucket_name:str=None,
            return_json=False,
            **kwargs
    ):
        """从 MinIO 存储中获取指定对象。
        该方法从预定义的 MinIO bucket 中获取指定的对象，并返回对象的内容。
        可以选择以 JSON 格式解析返回数据，或者直接返回二进制数据。
        Params:
            object_name(str): 对象存储的名称，形如: test_collection/test_doc.doc
            return_json(bool): 是否将对象内容作为 JSON 返回。如果为 True，将尝试将对象内容
                解码为 UTF-8 并解析为 JSON 格式。如果为 False，则返回原始二进制数据。
        Return:
            dict or bytes
            如果 `return_json` 为 True，则返回 JSON 解析后的字典；
            否则，返回对象的原始二进制数据。
        """
        bucket_name = bucket_name or self.config.bucket
        try:
            response = self.client.get_object(
                bucket_name=bucket_name,
                object_name=object_name,
                **kwargs
            )
            data = response.data
            if return_json:
                return json.loads(data.decode('utf-8'))
            return data
        finally:
            response.close()
            response.release_conn()

    def stat_object(
            self,
            object_name,
            **kwargs
    ):
        """获取指定对象的元数据,包括对象大小、最后修改时间、内容类型等信息。
        Params:
            object_name(str): 对象存储的名称，形如: test_collection/test_doc.doc
        Return:
            MinioObject: 返回包含对象元数据的对象实例。包括文件大小、修改时间、ETag 等信息。
        """
        stat = self.client.stat_object(
            bucket_name=self.config.bucket,
            object_name=object_name,
            **kwargs
        )
        return stat

    def get_metadata(self, object_name:str, bucket_name:str=None):
        """获取对象的元数据字典
        Params:
            object_name(str): 对象存储的名称，形如: test_collection/test_doc.doc
        Return:
            Dict: 对象的元数据字典
        """
        bucket_name = bucket_name or self.config.bucket
        stat = self.client.stat_object(
            bucket_name=bucket_name,
            object_name=object_name
        )
        metadata = stat.metadata
        pattern = 'x-amz-meta-'
        user_meta = dict()
        for k, v in metadata.items():
            if k.startswith(pattern):
                user_meta.update({k.strip(pattern): v})
        return user_meta

    def list_objects(
            self,
            bucket_name:str=None,
            prefix:str=None,
            recursive=False,
            return_name=True,
            **kwargs
    ):
        """列出指定路径前缀下的所有对象信息
        Params:
            prefix(str): 路径前缀，可以是完整路径。如果是目录，末尾要添加"/"符号
            recursive(bool): 是否递归展示，如果为False，只展示当前目录下的对象与目录信息
            return_name(bool): 是否以对象名形式返回
        Return:
            生成器，根据return_name参数，返回对象名称或对象信息
        """
        bucket_name = bucket_name or self.config.bucket
        objects = self.client.list_objects(
            bucket_name = bucket_name,
            prefix=prefix,
            recursive=recursive,
            **kwargs
        )
        for obj in objects:
            if return_name:
                yield obj.object_name
            else:
                yield obj

    def list_field_values(self, bucket_name:str=None, prefix:str=None, field='file_name', recursive=False):
        """列出对象元数据中"${field}"键对应的值
        Params:
            prefix(str): 对象路径，可以是文件的完整路径或目录路径
            field(str): 想要返回的文件元数据字段名称
            recursive(bool): 是否递归展示，如果为False，只展示当前目录下的对象与目录信息
        Return:
            生成器，返回路径下所有对象元数据中"${field}"键对应的值
        """
        bucket_name = bucket_name or self.config.bucket
        objects = self.client.list_objects(
            bucket_name,
            prefix=prefix,
            recursive=recursive
        )
        obj_names = [obj.object_name for obj in objects]
        func_filed_value = lambda obj: self.stat_object(obj).metadata.get(f'x-amz-meta-{field}')
        results = self.minio_pool.map(func_filed_value, obj_names, timeout=30)
        for field_value in results:
            if field_value:
                yield field_value

    def remove_object(
            self,
            bucket_name = None,
            prefix=None,
            object_list=None
    ):
        """删除对象，可以根据路径前缀删除，也可以根据对象名列表删除，两者选一。
        Params:
            bucket_name(str): 桶名
            prefix(str): 路径，可以是一个目录或文件，形如: test_collection/
            object_list(List[str]): 文件对象名列表，形如['test_collection/doc0.doc', 'test_collection/doc1.docx']
        """
        bucket_name = bucket_name or self.config.bucket
        if isinstance(prefix, list):
            object_list = prefix
            prefix = None
        if prefix:
            delete_object_list = map(
                lambda x: DeleteObject(x.object_name),
                self.client.list_objects(bucket_name, prefix, recursive=True),
            )
        elif object_list:
            delete_object_list = [
                DeleteObject(obj) for obj in object_list
            ]
        else:
            ...
        errors = self.client.remove_objects(bucket_name, delete_object_list)
        if errors:
            for error in errors:
                logger.error("error occurred when Minio deleting object", error)


    def generate_link(self, obj_name, bucket_name=None, expires_seconds=300):
        bucket_name = bucket_name or self.config.bucket
        return self.client.presigned_get_object(bucket_name, obj_name, expires=timedelta(seconds=expires_seconds))
            
        
MINIO_STORAGE = MinioStorage()

if __name__ == '__main__':
    # import io


    # file_path = '/home/kido/nlp/rag/cheap-RAG/LICENSE'
    # with open(file_path,'rb')as f:
    #     bytes_data = f.read()
    # data = io.BytesIO(bytes_data)
    # MINIO_STORAGE.put_object(object_name='my_test/LICENSE', data = data)
    # print('put successfully')

    # obj = MINIO_STORAGE.get_object(object_name='my_test/LICENSE')
    # obj = obj.decode()
    # print(obj[:100])

    # res = MINIO_STORAGE.remove_object(prefix='my_test/LICENSE')
    # print('object removed')
    ...
