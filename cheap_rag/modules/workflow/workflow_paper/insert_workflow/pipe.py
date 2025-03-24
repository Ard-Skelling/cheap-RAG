import io
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict
from pathlib import Path


# local module
from utils.logger import logger
from utils.helpers import SnowflakeIDGenerator, AsyncDict, atimer
from configs.config_cls import (
    LocalEmbeddingConfig,
    LlmConfig,
    WorkerConfig
)
from cheap_rag.modules.workflow.workflow_paper.config import (
    LOCAL_EMBEDDING_CONFIG,
    OCR_CONFIG,
    LLM_CONFIG,
    WORKER_CONFIG
)
from utils.tool_calling.api_calling import LlmApi, OcrApi
from utils.tool_calling.doc_processing import read_file
from utils.tool_calling.local_inferring.torch_inference import LocalEmbedding
from modules.storage import (
    MILVUS_STORAGE,
    ES_STORAGE,
    MINIO_STORAGE
)
from modules.workflow.workflow_paper.data_cls import (
    Task,
    TaskMeta
)
from modules.workflow.workflow_paper.db_opration import PAPER_KNOWLEDGE
from modules.workflow.workflow_paper.insert_workflow.workflow_tasks import (
    Chunking,
    InsertPreprocessing
)
from modules.task_manager import CoroTaskManager


class Worker:
    def __init__(
        self, 
        pool,
        semaphore,
        config: WorkerConfig = None
    ) -> None:
        """初始化工作者对象。它是执行具体业务逻辑的类。
        如果有流程的增加和功能的变化，建议在Worker中实现。
        适合处理 CPU + IO 密集型任务

        Params:
            pool(ProcessPoolExecutor): 进程池，用于在worker内的多进程任务
            config(WorkerConfig): 与InsertWorkflow使用同一套配置，
                主要是指定了进程池大小、异步队列大小等参数
        """
        self.config = config or WORKER_CONFIG
        self.pool = pool
        self.semaphore = semaphore
        self.task_queue = asyncio.Queue()
        self.task_db = dict()
        # 初始化工具调用
        self.init_tools()


    async def execute(self, task):
        async with self.semaphore:
            await self.task_queue.put(task)
            result = await self.inner_loop()
        return result


    def init_tools(self):
        self.ocr = OcrApi(OCR_CONFIG)
        self.llm = LlmApi(LLM_CONFIG)
        self.embedding = LocalEmbedding(LOCAL_EMBEDDING_CONFIG)


    @atimer
    async def standardize(self, task: Task):
        # TODO: Standardize input file
        raise NotImplementedError()


    @atimer
    async def to_ocr(self, task: Task):
        pdf_stem = Path(task.task_meta.file_name).stem
        pdf_bs64 = task.result
        await self.ocr.send_ocr(pdf_bs64, pdf_stem)
        task.task_meta.json_fp = self.ocr.config.ocr_cache \
            .joinpath(pdf_stem) \
            .joinpath(f'{pdf_stem}.json')
        task.step = 'chunking'
        # free pdf base64 memory
        task.task_meta.result = None
        return task


    @atimer
    async def chunking(self, task: Task):
        # Chunking任务相对复杂，用一个类来管理其下的处理逻辑
        chunker = Chunking(self.pool, task)
        file_name = task.task_meta.file_name
        json_fp = task.task_meta.json_fp
        doc = await read_file(str(json_fp))
        result = chunker.chunking(file_name, doc)
        task.result = result
        task.step = 'insert'
        logger.info(f'Chunking successfully: {file_name}')
        return task


    async def delete_file(self, task: Task):
        file_name = task.task_meta.file_name
        domain = task.task_meta.domain
        await PAPER_KNOWLEDGE.delete_file(domain, file_name)
        logger.info(f'Delete previous data successfully: {domain} - {file_name}')
        return task
    
    
    @atimer
    async def insert(self, task: Task):
        chunks = task.result['chunks']
        agg_chunks = task.result['agg_chunks']
        atom_chunks = task.result['atom_chunks']

        domain, file_name = task.task_meta.domain, task.task_meta.file_name
        
        preprocessor = InsertPreprocessing(self.pool, self.llm, self.embedding)
        chunks, aggs, atoms, emb_results = await preprocessor.process(file_name, chunks, agg_chunks, atom_chunks)
        
        # 在插入前，删除旧文件，确保没有重复文件
        if self.config.delete_old:
            task = await self.delete_file(task)

        tasks = []
        tasks.append(asyncio.to_thread(ES_STORAGE.bulk_insert_documents, f'{domain}_raw', chunks))
        tasks.append(asyncio.to_thread(ES_STORAGE.bulk_insert_documents, f'{domain}_agg', aggs))
        tasks.append(asyncio.to_thread(ES_STORAGE.bulk_insert_documents, f'{domain}_atom', atoms))
        tasks.append(asyncio.to_thread(MILVUS_STORAGE.insert, domain, emb_results))
        await asyncio.gather(*tasks)

        logger.info(f'Insert step finished: {file_name}')
        # 声明任务生命周期结束
        task.step = 'upload_object'
        task.result = None
        return task
    

    @atimer
    async def upload_object(self, task: Task):
        domain = task.task_meta.domain
        file_name = task.task_meta.file_name
        img_dir = self.ocr.config.ocr_cache.joinpath(Path(file_name).stem)
        imgs = img_dir.glob('*.jpg')
        to_upload = []
        for img in imgs:
            obj_name = f'{domain}/{file_name}/images/{img.name}'
            with open(str(img), 'rb') as f:
                obj_bytes = io.BytesIO(f.read())
            to_upload.append(asyncio.to_thread(
                MINIO_STORAGE.put_object,
                obj_name, 
                obj_bytes,
                MINIO_STORAGE.config.bucket_ocr
            ))
        await asyncio.gather(*to_upload)
        task.status = 'completed'
        task.result = 'success'
        task.step = 'FINISHED'
        return task


    def is_completed(self, task_id: int):
        task: Task = self.task_db[task_id]
        is_done = task.status in ['completed', 'failed']
        return is_done
    

    def free_cache(self, task:Task):
        file_name = task.task_meta.file_name
        raw_fp = self.config.raw_cache.joinpath(file_name)
        raw_fp.unlink(missing_ok=True)
        task.task_meta.pdf_fp.unlink(missing_ok=True)
    

    async def inner_loop(self):
        while True:
            try:
                await asyncio.sleep(0.1)    # 防止事件循环频繁get，提高CPU负荷
                # async with self.plock:
                task: Task = await self.task_queue.get()
                task.status = 'processing'
                # 获取worker中下一步处理流程的协程方法，并执行任务
                # 该方法只接收task作为参数，并返回task对象
                func = getattr(self, task.step, None)
                if func is None:
                    raise ValueError(f'Unsupported task step: {task.step}')
                if asyncio.iscoroutinefunction(func):
                    # 运行io密集型任务，如果任务耦合，可以在worker内解耦
                    task = await func(task)
                elif callable(func):
                    # 运行CPU密集型任务
                    task = func(task)
                else:
                    raise TypeError(f'Worker method must be function or coroutine.')
                # 告知异步队列当前环节任务完成
                # async with self.plock:
                self.task_queue.task_done()
                # 更新任务数据到self.task_db
                self.task_db[task.task_id] = task
                # 检测工作流是否完结（成功或失败），以决定是否继续下一流程
                if self.is_completed(task.task_id):
                    logger.info(
                        f'DataInsert task {task.task_id} done: {task.task_meta.file_name}\n'
                        f'Result: {task.result}'
                    )
                    # 将任务结果放入self.done_tasks，以便查阅
                    task = self.task_db.pop(task.task_id)
                    # 清理缓存
                    # self.free_cache(task)
                    # # 退出任务循环
                    return task.result
                else:
                    # async with self.plock:
                    await self.task_queue.put(task)
            except Exception as err:
                # TODO: 处理报错
                raise err


class InsertWorkflow:
    """RAG写入流程的工作流。
    一方面连接了负责写入数据处理逻辑的Worker类，另一方面连接了任务调度器TaskManager类。
    实现了写入工作流的异步多进程并行。"""
    def __init__(
        self, 
        config: WorkerConfig = None
    ) -> None:
        self.config = config or WORKER_CONFIG
        self.id_gen = SnowflakeIDGenerator(machine_id=0)
        self.semaphore = asyncio.Semaphore(self.config.num_semaphore)
        self.pending_tasks = AsyncDict(max_size=self.config.num_semaphore)
        # TODO: 使用redis来存储任务


    def init_tools(self):
        self.embedding = LocalEmbedding()
        self.llm = LlmApi()


    async def abstr_add_task(self, task_manager:CoroTaskManager, task_id, coro_func, *args, **kwargs):
        await task_manager.add_task(task_id, coro_func(*args, **kwargs))


    async def execute_task(self, pool, task_id, task_meta:TaskMeta):
        task = Task(
            task_id=task_id,
            task_meta=task_meta,
            step=task_meta.init_step,    # 设定任务起始环节
            status='pending',    # 设定任务起始状态
            result=task_meta.result
        )
        worker = Worker(
            pool,
            self.semaphore,
            self.config
        )
        result = await worker.execute(task)
        return task_id, result


    async def add_task(self, task_id, task_manager:CoroTaskManager, pool:ProcessPoolExecutor, task_meta:TaskMeta):
        await self.abstr_add_task(task_manager, task_id, self.execute_task, pool, task_id, task_meta)
        return task_id


    async def _submit(self, task_id, task_manager:CoroTaskManager, pool:ProcessPoolExecutor, task_meta:TaskMeta):
        await self.add_task(task_id, task_manager, pool, task_meta)
        result = await task_manager.wait_result(task_id, timeout=self.config.timeout)
        return result
    

    async def submit(self, task_manager:CoroTaskManager, pool:ProcessPoolExecutor, task_meta:TaskMeta):
        err = RuntimeError(f'{task_meta.file_name} max error retries.')
        for _ in range(5):
            try:
                task_id = self.id_gen.generate_id()
                await self.pending_tasks.put(task_id, task_meta)
                task_meta = await self.pending_tasks.pop(task_id)
                if not task_meta:
                    raise ValueError(f'Task {task_id} is not in ongoing tasks.')
                result = await self._submit(task_id, task_manager, pool, task_meta)
                return result
            except Exception as err:
                logger.warning(f'Error occurred: {task_meta.file_name}\n{repr(err)}')
                task_manager.tasks.pop(task_id, None)
                continue
        raise err
    

async def execute_one(
        workflow: InsertWorkflow, 
        task_manager: CoroTaskManager, 
        pool: ProcessPoolExecutor, 
        task_meta: TaskMeta, 
        log_path: Path
    ):
    file_name = task_meta.file_name
    try:
        result = await workflow.submit(task_manager, pool, task_meta)
        message = f'{file_name}: {result}'
        with open(str(log_path), 'a') as f:
            f.write(f'{message}\n\n')
    except Exception as err:
        message = f'{file_name}: {err}'
        with open(str(log_path), 'a') as f:
            f.write(f'{message}\n\n')
        raise err


async def main():
    raw_dir = Path('/root/nlp/rag/cheap-RAG/dev/output')
    log_dir = Path('/root/nlp/rag/cheap-RAG/dev/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir.joinpath('insert.log')

    tasks = []

    for cl in raw_dir.rglob('*_content_list.json'):
        f_stem = cl.stem.rstrip('_content_list')
        data_dir = cl.parent
        tasks.append(TaskMeta(
            domain='longevity_paper_2502', 
            file_name=f'{f_stem}.pdf',
            json_fp=cl,
            image_dir=data_dir.joinpath('images')
        ))


    task_manager = CoroTaskManager()
    flow = InsertWorkflow(WORKER_CONFIG)

    with ProcessPoolExecutor(6) as pool:
        semaphore = asyncio.Semaphore(8)
        async with semaphore:
            futures = []
            for task_meta in tasks:
                futures.append(
                    asyncio.create_task(
                        execute_one(
                            flow,
                            task_manager, 
                            pool, 
                            task_meta,
                            log_path
                        )
                    )
                )
            await asyncio.gather(*futures)

    print('All tasks are finished.')


if __name__ == '__main__':
    asyncio.run(main())


