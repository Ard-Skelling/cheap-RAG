import asyncio
from typing import Union, List


# Local modules
from utils.logger import logger
from modules.storage import (
    MILVUS_STORAGE,
    ES_STORAGE,
    MINIO_STORAGE
)
from modules.workflow.workflow_paper.storage_design import (
    FIELDS,
    INDEX_SETTINGS,
    ES_SETTINGS
)





class PaperKnowledge:
    async def create_db(self, domains: Union[str, List[str]]):
        """Create data containers for certain domain knowledge bases.

        Args:
            domains (Union[str, List[str]]): name(s) for domain knowledge base(s).
        """
        if isinstance(domains, str):
            domains = [domains]
        tasks = []
        for domain in domains:
            # Create ES index
            for es_setting in ES_SETTINGS:
                suffix = es_setting['suffix']
                mappings = es_setting['mappings']
                create_es = asyncio.to_thread(
                    ES_STORAGE.create_index,
                    index_name=f'{domain}_{suffix}',
                    mappings=mappings
                )
                tasks.append(create_es)
            # Create Milvus collections
            create_mil = asyncio.to_thread(
                MILVUS_STORAGE.create_collection,
                collection_name=domain,
                fields=FIELDS,
                index_settings=INDEX_SETTINGS
            )
            tasks.append(create_mil)
            await asyncio.gather(*tasks)
        logger.info(f'Create domain knowledge seccessfully: {domains}')

    
    async def drop_db(self, domains: Union[str, List[str]]):
        """Drop data containers for certain domain knowledge bases.

        Args:
            domains (Union[str, List[str]]): name(s) for domain knowledge base(s).
        """
        if isinstance(domains, str):
            domains = [domains]
        tasks = []
        for domain in domains:
            # Drop ES index
            for setting in ES_SETTINGS:
                drop_es = asyncio.to_thread(
                    ES_STORAGE.delete_index,
                    index_name=f"{domain}_{setting['suffix']}"
                )
                tasks.append(drop_es)
            # Drop Milvus collections
            drop_mil = asyncio.to_thread(
                MILVUS_STORAGE.drop_collection,
                collection_name=domain
            )
            tasks.append(drop_mil)
            await asyncio.gather(*tasks)
        logger.info(f'Drop domain knowledge seccessfully: {domains}')


    async def delete_file(self, domain: str, file_names: Union[str, List[str]]):
        """Delete file datas by file name(s).

        Args:
            domain (str): domain knowledge name
            file_names (Union[str, List[str]]): file(s) to delete.
        """
        
        if isinstance(file_names, str):
            file_names = [file_names]
        tasks = []
        # Delete ES files
        for setting in ES_SETTINGS:
            del_es = asyncio.to_thread(
                ES_STORAGE.delete_doc,
                index_name=f"{domain}_{setting['suffix']}",
                query={"terms": {"file_name": file_names}}
            )
            tasks.append(del_es)
        # Delete Milvus files
        del_mil = asyncio.to_thread(
            MILVUS_STORAGE.delete,
            collection_name=domain,
            filter=f'file_name in {file_names}'
        )
        tasks.append(del_mil)
        # Delete Minio files
        minio_dirs = [f'{domain}/{f}' for f in file_names]
        for obj_prefix in minio_dirs:
            for bucket in [
                MINIO_STORAGE.config.bucket,
                MINIO_STORAGE.config.bucket_ocr
            ]:
                del_minio = asyncio.to_thread(
                    MINIO_STORAGE.remove_object,
                    bucket_name=bucket,
                    prefix=obj_prefix
                )
                tasks.append(del_minio)
        await asyncio.gather(*tasks)
        logger.info(f'Delete domain knowledge files seccessfully: {domain} - count: {len(file_names)}')


    async def list_files(self, domain: str):
        """List all files in knowledge base.

        Args:
            domain (str): knowledge base name.
        """
        result = await asyncio.to_thread(ES_STORAGE.search_unique, index_name=f'{domain}_raw', field='file_name')
        return result


PAPER_KNOWLEDGE = PaperKnowledge()



if __name__ == '__main__':
    domain = 'longevity_paper_2502'
    asyncio.run(PAPER_KNOWLEDGE.create_db(domain))
    # res = MILVUS_STORAGE.describe_collection(domain)
    # print(res)
    # asyncio.run(PAPER_KNOWLEDGE.drop_db(domain))