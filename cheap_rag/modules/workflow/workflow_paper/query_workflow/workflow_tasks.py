import asyncio
from typing import List, Tuple
from collections import defaultdict


# Local modules
from utils.helpers import generate_md5
from modules.storage import MILVUS_STORAGE, ES_STORAGE, MINIO_STORAGE
from utils.tool_calling.local_inferring.onnx_inference import LocalEmbedding
from modules.workflow.workflow_paper.config import QUERY_EMBEDDING_CONFIG


class HybridSearch:
    def __init__(self):
        self.embedding = LocalEmbedding(QUERY_EMBEDDING_CONFIG)
        self.task_map = {
            ''
        }
    

    @staticmethod
    def cross_sort(milvus_resp, es_resp):
        # 在穿插排序的同时进行去重
        result = dict()
        m_len = len(milvus_resp)
        e_len = len(es_resp)
        while m_len + e_len > 0:
            if m_len > 0:
                m_rec = milvus_resp.pop(0)
                result[m_rec['chunk_id']] = m_rec
                m_len -= 1
            if e_len > 0:
                e_rec = es_resp.pop(0)
                if e_rec['chunk_id'] not in result:
                    result[e_rec['chunk_id']] = e_rec
                e_len -= 1
        return list(result.keys())
    

    async def get_chunks(self, domain, chunk_ids: List[int], output_fields:List[str]):
        """通过原子文本块的唯一标识doc_id，
        查询出其对应的聚合文本块，并保证相关度排序的一致性。
        """
        order = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
        query = {'terms': {'chunk_id': chunk_ids}}
        res = await asyncio.to_thread(
            ES_STORAGE.search_documents, 
            f'{domain}_raw', 
            query, 
            output_fields, 
            size=0
        )
        order_rec = {}
        for rec in res:
            # order_rec = {序号: 数据字典}
            order_rec[order[rec['chunk_id']]] = rec
        res = [order_rec[key] for key in sorted(order_rec)]
        return res
    

    async def parallel_recall(self, domain:str, output_fields:list, chunks:List[dict]):
        """并行召回图表片段的上下片

        Params:
            recalling_records(Dict[dict]): 待召回的数据，形如：
                {
                    0: {
                        "file_name": "xxx.pdf",
                        "chunk_index": 45
                    }
                }
            dup_set(set): 已收录的数据文本片段前100个字符的md5 hash，用于数据去重
        """
        body = []
        ids = []

        build_query = lambda file_name, index: {
            'query': {
                'bool': {
                    'must': [
                        {'term': {'file_name': file_name}},
                        {'term': {'chunk_index': int(index)}},
                    ]
                }
            }
        }

        chunks = {i: chunk for i, chunk in enumerate(chunks)}

        for count, record in chunks.items():
            ids.append(count)
            body.append(dict())
            body.append(build_query(record['file_name'], record['chunk_index'] - 1))
            body.append(dict())
            body.append(build_query(record['file_name'], record['chunk_index']))
            body.append(dict())
            body.append(build_query(record['file_name'], record['chunk_index'] + 1))
        
        if not body:
            return list(chunks.values())
        # 发起并发多重查询，output_fields和domain来自于上文
        es_resp = ES_STORAGE.multi_search(body, output_fields, f'{domain}_raw', is_flatten=False)
        # es_resp 形如 [
        #     [{...}],    前一片文本
        #     [],    当前片文本
        #     [{...}],    后一片文本
        # ]
        return es_resp
    



    
    async def vector_search(
        self, 
        domain:str, 
        query:str, 
        topk: int,
        output_fields:list, 
        threshold:float,
        task_name='MedicalRetrieval'
    ):
        query = self.embedding.build_query(query, task_name=task_name)
        query_vec = await self.embedding.a_embedding(query)
        mil_res = await asyncio.to_thread(
            MILVUS_STORAGE.search,
            domain,
            query_vec.tolist(),
            limit=topk,
            output_fields=output_fields,
        )
        final_res = []
        for res in mil_res[0]:
            if res.get('distance') >= threshold:
                # Sample res:
                # {
                #     'doc_id': '1', 
                #     'distance': 0.8519943356513977, 
                #     'entity': {'color': 'red_7025'}
                # }
                final_res.append(res)
        return final_res


    async def bm25_search(
        self, 
        domain:str, 
        query:str, 
        topk: int,
        output_fields:list
    ):
        es_query = {"match": {"text": query}}
        es_res = await asyncio.to_thread(
            ES_STORAGE.search_documents,
            f'{domain}_atom',
            es_query,
            output_fields,
            topk
        )
        return es_res


    async def search(
        self, 
        query:str, 
        domain:str, 
        topk: int = 10, 
        output_fields: list = None,
        threshold:float=0.4,
        task_name:str='MedicalRetrieval'
    ):
        # 需要用file_name和agg_index映射数据的顺序，确保这两个字段一定在输出字段中
        output_fields = output_fields or ['text', 'caption', 'footnote', 'page_index', 'chunk_type', 'url', 'text_level']
        must_has = {'chunk_index', 'file_name'}
        output_fields = list(set(output_fields).union(must_has))
        try:
            # Vector + BM25 search
            search_tasks = [
                self.vector_search(domain, query, topk, ['chunk_id'], threshold, task_name),
                self.bm25_search(domain, query, topk, ['chunk_id'])
            ]
            mil_res, es_res = await asyncio.gather(*search_tasks)
            mil_res = [m['entity'] for m in mil_res]

            # es + milvus crossing sort
            chunk_ids = self.cross_sort(mil_res, es_res)

            # get chunks
            chunks = await self.get_chunks(domain, chunk_ids, output_fields=['chunk_id', 'chunk_index', 'file_name'])

            # get context
            chunks = await self.parallel_recall(domain, output_fields, chunks)
            
            return chunks

        except Exception as err:
            raise err
    

    
    async def query(self):
        # TODO: database query method
        ...


if __name__ == '__main__':

    domain = 'longevity_paper_2502'

    query = 'BrainAGE'


    searher = HybridSearch()
    res = asyncio.run(searher.search(query, domain))
    ...