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
    

    @staticmethod
    def cross_sort(milvus_resp, es_resp):
        # Wave vector and BM25 search result together.
        # Use (agg_index, file_name) as the key of result and drop the duplicated.
        result = dict()
        m_len = len(milvus_resp)
        e_len = len(es_resp)
        while m_len + e_len > 0:
            if m_len > 0:
                m_rec = milvus_resp.pop(0)
                result[(m_rec['agg_index'], m_rec['file_name'])] = m_rec
                m_len -= 1
            if e_len > 0:
                e_rec = es_resp.pop(0)
                if (e_rec['agg_index'], e_rec['file_name']) not in result:
                    result[(e_rec['agg_index'], e_rec['file_name'])] = e_rec
                e_len -= 1
        return list(result.keys())
    

    async def get_chunks(self, domain, index_name_pair: List[Tuple[int, str]], output_fields:List[str]):
        """Query the agg_chunks with """
        if not index_name_pair:
            return []
        order = {index_name: i for i, index_name in enumerate(index_name_pair)}
        build_query = lambda agg_index, file_name: {
            'query': {
                'bool': {
                    'must': [
                        {'term': {'file_name': file_name}},
                        {'term': {'agg_index': int(agg_index)}},
                    ]
                }
            }
        }
        order = dict()
        body = []
        for i, (index, name) in enumerate(index_name_pair):
            order[(index, name)] = i
            body.append(dict())
            body.append(build_query(index, name))
        es_resp = await asyncio.to_thread(
            ES_STORAGE.multi_search,
            body=body,
            output_fields=output_fields,
            index_name=f'{domain}_agg',
            is_flatten=False
        )
        return es_resp
    

    async def parallel_recall(self, domain:str, output_fields:list, chunks:List[dict]):
        """Recall the previous, current and next chunks parallelly"""
        body = []

        build_query = lambda file_name, index: {
            'query': {
                'bool': {
                    'must': [
                        {'term': {'file_name': file_name}},
                        {'term': {'agg_index': int(index)}},
                    ]
                }
            }
        }

        for record in chunks:
            body.append(dict())
            body.append(build_query(record['file_name'], record['agg_index'] - 1))
            body.append(dict())
            body.append(build_query(record['file_name'], record['agg_index']))
            body.append(dict())
            body.append(build_query(record['file_name'], record['agg_index'] + 1))
        
        # Launch ElasticSearch multi-query with query body
        es_resp = await asyncio.to_thread(
            ES_STORAGE.multi_search,
            body=body,
            output_fields=output_fields,
            index_name=f'{domain}_agg',
            is_flatten=False
        )
        # es_resp example:
        # [
        #     [{...}],    previous agg_chunk
        #     [],    current agg_chunk
        #     [{...}],    next agg_chunk
        # ]
        es_resp = [e[0] for e in es_resp if e]
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


    async def replace_file_name(self, domain:str, results:List[dict]):
        file_name_map = dict()
        for record in results:
            hash_name = record['file_name']
            if hash_name in file_name_map:
                file_name = file_name_map[hash_name]
            else:
                query = {
                    'bool': {'must': [
                        {'term': {'file_name': hash_name}},
                        {'term': {'chunk_type': 'title'}}
                    ]}
                }
                result = await asyncio.to_thread(ES_STORAGE.search_documents, f'{domain}_raw', query, output_fields=['text'], size=1)
                file_name = result[0]['text'].strip()
                file_name_map[hash_name] = file_name
            record['file_name'] = file_name
        return results
    

    async def search(
        self, 
        query:str, 
        domain:str, 
        topk: int = 10, 
        output_fields: list = None,
        threshold:float=0.4,
        task_name:str='MedicalRetrieval',
        has_context:bool=False
    ):
        # Must has file_name and agg_index fields，which are used to keep similarity sorted
        output_fields = output_fields or ['text', 'caption', 'footnote', 'page_index', 'chunk_type', 'url', 'text_level']
        must_has = {'chunk_index', 'file_name'}
        output_fields = list(set(output_fields).union(must_has))
        try:
            # Vector + BM25 search
            search_tasks = [
                self.vector_search(domain, query, topk, ['agg_index', 'file_name'], threshold, task_name),
                self.bm25_search(domain, query, topk, ['agg_index', 'file_name'])
            ]
            mil_res, es_res = await asyncio.gather(*search_tasks)
            mil_res = [m['entity'] for m in mil_res]

            # es + milvus crossing sort
            index_name_pairs = self.cross_sort(mil_res, es_res)

            # get chunks
            chunks = await self.get_chunks(domain, index_name_pairs, output_fields=['agg_index', 'page_index', 'text', 'file_name', 'chunk_type'])
            if not chunks:
                return []
            chunks = [chunk[0] for chunk in chunks]
            # get context
            if has_context:
                chunks = await self.parallel_recall(domain, output_fields, chunks)
            
            chunks = await self.replace_file_name(domain, chunks)
            return chunks

        except Exception as err:
            raise err
    

    
    async def query(self):
        # TODO: database query method
        ...


if __name__ == '__main__':

    domain = 'longevity_paper_2502'

    query = "How to evaluate the brain's age?"


    searher = HybridSearch()
    res = asyncio.run(searher.search(query, domain, has_context=False))
    ...