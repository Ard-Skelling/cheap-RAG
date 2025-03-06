import asyncio
from typing import List
from functools import wraps


# Local modules
from configs.config_cls import QueryConfig
from modules.workflow.workflow_paper.query_workflow.workflow_tasks import HybridSearch
from modules.workflow.workflow_paper.config import QUERY_CONFIG


class QueryWorkflow:
    def __init__(self, config: QueryConfig = None):
        self.config = config or QUERY_CONFIG
        self.init_searchers()
        self.semaphore = asyncio.Semaphore(self.config.semaphore)
        

    def init_searchers(self):
        self.hybrid_searcher = HybridSearch()


    def with_semaphore(coro):
        @wraps(coro)
        async def wrapper(self, *args, **kwargs):
            async with self.semaphore:
                result = await coro(self, *args, **kwargs)
            return result
        return wrapper


    @with_semaphore
    async def search(
        self,
        query: str,
        domain: str,
        topk: int = 10,
        output_fields: List[str] = None,
        threshold: float = 0.4,
        task_name: str = 'MedicalRetrieval',
        has_context: bool = False,
        filter: dict = None
    ):
        """_summary_

        Args:
            query (str): The query string, such as a question, statement, etc.
            domain (str): Knowledge base name.
            topk (int, optional): Top k chunks to retrieve in 
                each of vector and BM25 dataset. Defaults to 10.
            output_fields (List[str], optional): Output fields in result. Defaults to None.
            threshold (float, optional): Threshold for vector search similarity. Defaults to 0.4.
            task_name (str, optional): 
                Refer to cheap_rag/utils/tool_calling/local_inferring/embedding_utils.py. 
                Defaults to 'MedicalRetrieval'.
            has_context (bool, optional): Return the previous and next chunk for retrieved chunk.
                Defaults to False.
            filter (dict, optional): The filter conditions in vector and BM25 search. 
                Defaults to None.

        Returns:
            List[dict]: Retrieved aggregated chunks, sorted by similarity.
        """
        # TODO: apply filter in search
        result = await self.hybrid_searcher.search(
            query,
            domain,
            topk=topk,
            output_fields=output_fields,
            threshold=threshold,
            task_name=task_name,
            has_context=has_context
        )
        return result
    

    def query(self):
        # TODO: Normal query or aggregated query.
        raise NotImplementedError('Unfinished method.')
    

if __name__ == '__main__':
    domain = 'longevity_paper_2502'

    query = "How to evaluate the brain's age?"

    workflow = QueryWorkflow()
    res = asyncio.run(workflow.search(query, domain))
    print(len(res))