import sys
import asyncio
import re
import logging
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Dict
from pathlib import Path


BASE_PATH = Path(__file__).parent.parent.parent.parent
sys.path.append(str(BASE_PATH))


# local module
from common.utils import SnowflakeIDGenerator, AsyncDict, atimer
from configs.vector_database_config import GLOBAL_CONFIG
from configs.config_v2.config_cls import WorkerConfig
from configs.config_v2.config import WORKER_CONFIG
from configs.config_v2.config_cls import FileConvertConfig
from module.pipeline_v2.tool_calling import (
    FileConverter, 
    OcrApi,
    LlmApi,
    EmbeddingApi,
    read_file
)
from module.pipeline_v2.data_cls import (
    Task,
    TaskMeta,
    GraphInfo,
    AggChunk,
    AtomChunk,
    TableChunk,
    MilvusDataV2
)
from module.pipeline_v2.task_manager import CoroTaskManager
from module.pipeline_v2.insert_workflow.workflow_tasks import (
    Chunking,
    InsertPreprocessing
)
from module.vecdatas.predict import PyMilvusInference
from module.storage.elastic_storage import ES_STORAGE


# TODO: 启用新的Milvus存储对象
MILVUS_STORAGE = PyMilvusInference(GLOBAL_CONFIG.vecdata_config)


class Worker:
    """异步多进程的数据处理流程，实现了多文件处理的异步并行。
    1. TODO: 文件格式转化（尚未纳入并行，需启用异步接口，在全局层面实现异步并行）
    2. TODO: OCR（尚未纳入并行，同上）
    3. OCR结果划分任务子集
    4. 对于每个子集，异步并行执行以下操作
        4.1 chunk规整，并写入chunk字典
        4.2 chunk切分子句，识别文本、表格和图片
            4.2.1 文本写入atom_chunk字典
            4.2.2 图片暂不处理
            4.2.3 表格处理如下
                4.2.3.1 生成 question * 2
                4.2.3.2 构造 question * 2
                4.2.3.3 qa 对写入atom_chunk字典
        4.3 构造 embedding batch，生成embedding，写入向量库
    5. 归约子集数据，形成agg_chunks，chunks和atom_chunks三级数据，
        展平写入ES。TODO: 分表存储，减少查询开销
    6. 清理缓存，释放内存
    7. TODO: 错误捕获并回滚失败任务，删除已写入数据
    """
        

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
        # self.done_tasks = done_tasks
        # 初始化外部工具调用
        self.init_tools()


    async def execute(self, task):
        async with self.semaphore:
            await self.task_queue.put(task)
            result = await self.inner_loop()
        return result


    def init_tools(self):
        convertor_config = FileConvertConfig(
            num_workers=self.config.num_workers
        )
        self.file_convertor = FileConverter(convertor_config)
        self.ocr = OcrApi()
        self.llm = LlmApi()
        self.embedding = EmbeddingApi()


    @atimer
    async def convert(self, task: Task):
        raw_file_dir = self.file_convertor.config.raw_cache
        out_put_dir = self.file_convertor.config.convert_cache
        file_name = task.task_meta.file_name
        raw_fp = raw_file_dir.joinpath(file_name)
        suffix = raw_fp.suffix.lower()
        # 在task_meta中记录文件本地路径
        if suffix == '.pdf':
            task.task_meta.pdf_fp = raw_fp
        elif suffix in ['.doc', '.docx']:
            task.task_meta.pdf_fp = out_put_dir.joinpath(file_name).with_suffix('.pdf')
            _ = await self.file_convertor.a_convert_file(str(raw_fp), str(out_put_dir), 'pdf')
            # 移除旧文件缓存
            # raw_fp.unlink()
        else:
            # TODO: 处理其它类型的文件
            raise ValueError(f'Unsupported file format: {suffix}. Please submit .doc, .docx or .pdf')
        # 设定下一步处理流程
        task.step = 'to_ocr'
        print(f'Convert file successfully: {file_name}')
        return task


    @atimer
    async def to_ocr(self, task: Task):
        pdf_fp = task.task_meta.pdf_fp
        while not pdf_fp.exists():
            await asyncio.sleep(0.5)
        pdf_prefix = pdf_fp.stem
        domain = task.task_meta.domain
        pdf_bytes = read_file(str(pdf_fp))
        ocr_res = await self.ocr.send_ocr(pdf_bytes, pdf_prefix, domain)
        # Debug模式下，保存ocr结果到本地，便于分析
        if GLOBAL_CONFIG.debug_mode:
            fp = self.ocr.config.ocr_cache.joinpath(f'{pdf_prefix}.json')
            with open(str(fp), 'w', encoding='utf-8') as f:
                json.dump(ocr_res, f, indent=4, ensure_ascii=False)
        # 将ocr结果暂存到task对象的result字段中，以便下一步使用
        task.result = ocr_res
        # 设定下一步处理流程
        task.step = 'chunking'
        print(f'OCR successfully: {pdf_fp.name}')
        return task


    @atimer
    async def chunking(self, task: Task):
        # Chunking任务相对复杂，用一个类来管理其下的处理逻辑
        chunker = Chunking(self.pool, task)
        file_name = task.task_meta.file_name
        doc_key = str(Path(file_name).with_suffix('.md'))
        # 获取待处理的OCR文档正文markdown
        doc = task.result.pop(doc_key)
        result = chunker.chunking(doc)
        task.result = result
        task.step = 'insert'
        print(f'Chunking successfully: {file_name}')
        return task


    @atimer
    async def delete(self, task: Task):
        file_name = task.task_meta.file_name
        domain = task.task_meta.domain
        del_mil = asyncio.to_thread(MILVUS_STORAGE._delete_data, [{"file_name": file_name}], domain)
        del_es = asyncio.to_thread(ES_STORAGE.delete_doc, domain, query={"terms": {"file_name": [file_name]}})
        await asyncio.gather(*[del_mil, del_es])
        print(f'Delete previous data successfully: {domain} - {file_name}')
        return task

    
    @atimer
    async def insert(self, task: Task):
        agg_chunks = task.result['agg_chunks']
        atom_chunks = task.result['atom_chunks']

        domain, file_name = task.task_meta.domain, task.task_meta.file_name
        
        preprocessor = InsertPreprocessing(self.pool, self.llm, self.embedding)
        aggs, text_atoms, table_atoms, emb_results = await preprocessor.process(domain, file_name, agg_chunks ,atom_chunks)
        
        # 在插入前，删除旧文件，确保没有重复文件
        task = await self.delete(task)

        tasks = []
        to_es = aggs + text_atoms + table_atoms
        tasks.append(asyncio.to_thread(ES_STORAGE.bulk_insert_documents, domain, to_es))
        tasks.append(asyncio.to_thread(MILVUS_STORAGE._insert_data, emb_results, domain, version=2))
        await asyncio.gather(*tasks)

        print(f'Insert step finished: {file_name}')
        # 声明任务生命周期结束
        task.status = 'completed'
        task.result = 'success'
        task.step = 'FINISHED'
        return task


    def is_completed(self, task_id: int):
        task: Task = self.task_db[task_id]
        is_done = task.status in ['completed', 'failed']
        return is_done
    

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
                    print(
                        f'DataInsert task {task.task_id} done: {task.task_meta.file_name}\n'
                        f'Result: {task.result}'
                    )
                    # 将任务结果放入self.done_tasks，以便查阅
                    # 外层函数会轮询done_tasks检查任务状态，返回执行情况。

                    #
                    # with self.plock:
                    task = self.task_db.pop(task.task_id)
                    # 清理缓存
                    file_name = task.task_meta.file_name
                    raw_fp = self.config.raw_cache.joinpath(file_name)
                    raw_fp.unlink(missing_ok=True)
                    task.task_meta.pdf_fp.unlink(missing_ok=True)
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


    async def abstr_add_task(self, task_manager:CoroTaskManager, task_id, coro_func, *args, **kwargs):
        await task_manager.add_task(task_id, coro_func(*args, **kwargs))


    async def execute_task(self, pool, task_id, task_meta):
        task = Task(
            task_id=task_id,
            task_meta=task_meta,
            step='convert',    # 设定任务起始环节
            status='pending'    # 设定任务起始状态
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
        task_id = self.id_gen.generate_id()
        await self.pending_tasks.put(task_id, task_meta)
        err = RuntimeError(f'{task_meta.file_name}超过最大重试次数')
        for _ in range(5):
            try:
                task_meta = await self.pending_tasks.pop(task_id)
                if not task_meta:
                    raise ValueError(f'Task {task_id} is not in ongoing tasks.')
                result = await self._submit(task_id, task_manager, pool, task_meta)
                return result
            except Exception as err:
                logging.warning(f'Error occurred: {task_meta.file_name}\n{repr(err)}')
                task_manager.tasks.pop(task_id, None)
                continue
        raise err
    

if __name__ == '__main__':
    ...


    



    