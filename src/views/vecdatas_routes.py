# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import io
import json
import time
import re
import logging
import traceback
import threading
from collections import defaultdict
from pathlib import Path
from flask import Response, Blueprint
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple


# local module
from configs import PublicEngineConfig
from configs.vector_database_config import GLOBAL_CONFIG
from configs.config_v2.config import WORKER_CONFIG
from app_core import init_vecdata_engine
from common import cost_time, VecDataApiCommon, JSONEncoder, generate_md5, send_request_rerank_server
from common.deal_response import response_func, BadRequest
from module.storage.elastic_storage import ES_STORAGE
from app_milvus_server.module.pipeline_v2.task_manager import CoroTaskManager
from app_milvus_server.module.pipeline_v2.insert_workflow.pipe import InsertWorkflow, Worker
from module.pipeline_v2.data_cls import TaskMeta


# 注册向量模块视图
vecdata_blue = Blueprint("vecdatas", __name__)
# 初始化推理引擎
vecDataEngine = init_vecdata_engine()
# 创建当前模块的写线程池, 保证每次只有一个任务在运行
write_thread_pool = ThreadPoolExecutor(max_workers=2)
# 创建当前模块的读线程池, 限制查询最大并发量
read_thread_pool = ThreadPoolExecutor(max_workers=16)
# 创建重排模型线程池
rerank_pool = ThreadPoolExecutor(max_workers=GLOBAL_CONFIG.rerank_config['thread_num'])
# 全局进程池
GLOBAL_POOL = ProcessPoolExecutor(WORKER_CONFIG.num_workers)


# 任务管理器，用于异步多进程并行
TASK_MANAGER = CoroTaskManager()

# def start_monitoring():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.create_task(TASK_MANAGER.monitor_tasks())


# monitor = threading.Thread(target=start_monitoring)
# monitor.start()

INSERT_WF = InsertWorkflow()


def construct_response_result(result, csid):
    """
        构造响应体
    """
    final_result = {}
    final_result["code"] = "200"
    final_result["messages"] = "success"
    final_result["version"] = PublicEngineConfig.VERSION
    final_result["csid"] = csid
    final_result["timestamp"] = int(time.time())
    final_result["result"] = result

    return json.dumps(final_result, ensure_ascii=False, cls=JSONEncoder)

# 获取数据库接口
@vecdata_blue.route("/v1/model/VecDataEngine", methods=["POST"])
def vecDataEntityServer():
    """"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        csid = VecDataApiCommon.parse_vecdata_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))

        future = read_thread_pool.submit(vecDataEngine.get_vecdata_result, csid)
        response = future.result()
        
        return Response(construct_response_result(response, csid=csid), mimetype="application/json")
    except Exception as err:
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")


# 数据库创建接口
@vecdata_blue.route("/v1/model/VecDataCreateEngine", methods=["POST"])
def vecDataCreateServer():
    """"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        collections, csid, version = VecDataApiCommon.parse_vecdata_create_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))

        future = write_thread_pool.submit(vecDataEngine.create_vecdata, collections, csid, version=version)
        response = future.result()
        return Response(construct_response_result(response, csid=csid), mimetype="application/json")
    except Exception as err:
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")


# 数据库删除接口
@vecdata_blue.route("/v1/model/VecDataDeleteEngine", methods=["POST"])
def vecDataDeleteServer():
    """"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        collections, csid = VecDataApiCommon.parse_vecdata_delete_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))

        future = write_thread_pool.submit(vecDataEngine.delete_vecdata, collections, csid)
        response = future.result()
        
        return Response(construct_response_result(response, csid=csid), mimetype="application/json")

    except Exception as err:
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")


# 数据库插入数据
@vecdata_blue.route("/v1/model/DataInsertEngine", methods=["POST"])
async def dataInsertServer():
    """数据插入"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        document, domain, file_name, file_meta, csid, is_enhanced, version = VecDataApiCommon.parse_data_insert_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))
        suffix = Path(file_name).suffix.lower()
        if version == 1 or suffix in {'.xlsx', '.xls', '.txt', '.json', '.csv'}:
            future = write_thread_pool.submit(vecDataEngine.insert_data, document, domain, file_name, file_meta, csid, is_enhanced=is_enhanced, version=version)
            response = future.result()

        elif version == 2 and suffix in {'.pdf', '.doc', '.docx'}:
            # 存储文件到缓存目录
            bytes_data = base64.b64decode(document.encode("utf-8"))
            cache_dir = INSERT_WF.config.raw_cache
            fp = cache_dir.joinpath(file_name)
            with open(str(fp), 'wb') as f:
                f.write(bytes_data)

            # 做成任务
            task_meta = TaskMeta(
                domain=domain,
                file_name=file_name,
                csid=csid
            )
            response = await INSERT_WF.submit(TASK_MANAGER, GLOBAL_POOL, task_meta)
        
        return Response(construct_response_result(response, csid=csid), mimetype="application/json")

    except Exception as err:
        logging.error(traceback.format_exc())
        err = BadRequest(response_func(csid=csid, code='10003', message=repr(err)))
        return Response(err.message, mimetype="application/json")


# 数据库删除数据
@vecdata_blue.route("/v1/model/DataDeleteEngine", methods=["POST"])
def dataDeleteServer():
    """数据删除"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        domain, file_name, csid = VecDataApiCommon.parse_data_delete_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))

        future = write_thread_pool.submit(vecDataEngine.delete_data, domain, file_name, csid)
        response = future.result()
        
        return Response(construct_response_result(response, csid=csid), mimetype="application/json")

    except Exception as err:
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")


# 数据查询接口
@vecdata_blue.route("/v1/model/DataSearchEngine", methods=["POST"])
def DataSearchServer():
    """数据检索服务"""
    # TODO: 剥离嵌套函数，写入独立的查询逻辑模块
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        question, domain, search_field, output_fields, threshold, topn, csid, use_rerank, version = VecDataApiCommon.parse_data_search_request()
        output_fields = list(set(output_fields).union({'type', 'index', 'file_name'}))
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))

        def retrieve_context(index, record):
            rec = vecDataEngine.scaler_query_data(
                question=[index, record['file_name'], 'text'],
                domain=domain,
                search_field=['index', 'file_name', 'type'],
                topn=1,
                csid=csid,
                output_fields=output_fields,
                operator=['==', '==', '==']
            )
            return rec

        def es_retrieve_context(index, record):
            query = {
                'bool': {
                    'must': [
                        {'term': {'file_name': record['file_name']}},
                        {'term': {'index': int(index)}},
                        {'term': {'chunk_type': 'text'}}
                    ]
                }
            }
            res = ES_STORAGE.search_documents(domain, query, output_fields=output_fields)
            return res
        
        def cross_sort(milvus_resp, es_resp, version=1):
            if version == 1:
                result = []
                m_len = len(milvus_resp)
                e_len = len(es_resp)
                while m_len + e_len > 0:
                    if m_len > 0:
                        result.append(milvus_resp.pop(0))
                        m_len -= 1
                    if e_len > 0:
                        result.append(es_resp.pop(0))
                        e_len -= 1
                return result
            if version == 2:
                # 在穿插排序的同时进行去重
                result = dict()
                m_len = len(milvus_resp)
                e_len = len(es_resp)
                while m_len + e_len > 0:
                    if m_len > 0:
                        m_rec = milvus_resp.pop(0)
                        result[m_rec['doc_id']] = m_rec
                        m_len -= 1
                    if e_len > 0:
                        e_rec = es_resp.pop(0)
                        result[e_rec['doc_id']] = e_rec
                        e_len -= 1
                return list(result.keys())


        def parallel_recall(recalling_records:dict, dup_set:set):
            """并行召回图表片段的上下片

            Params:
                recalling_records(Dict[dict]): 待召回的图表型数据，形如：
                    {
                        0: {
                            "type": "table",
                            "data": {...}
                        }
                    }
                dup_set(set): 已收录的数据文本片段前100个字符的md5 hash，用于数据去重
            """
            body = []
            graph_ids = []
            build_query = lambda file_name, index: {
                'query': {
                    'bool': {
                        'must': [
                            {'term': {'file_name': file_name}},
                            {'term': {'index': int(index)}},
                            {'term': {'chunk_type': 'text'}}
                        ]
                    }
                }
            }
            for count, record in recalling_records.items():
                if record['type'] == 'table' or record['type'] == 'image':
                    graph_ids.append(count)
                    body.append(dict())
                    body.append(build_query(record['file_name'], record['index'] - 1))
                    body.append(dict())
                    body.append(build_query(record['file_name'], record['index']))
                    body.append(dict())
                    body.append(build_query(record['file_name'], record['index'] + 1))
            
            if not body:
                return list(recalling_records.values())
            # 发起并发多重查询，output_fields和domain来自于上文
            graph_id = -1
            es_resp = ES_STORAGE.multi_search(body, output_fields, domain, is_flatten=False)
            # es_resp 形如 [
            #     [{...}],    前一片文本
            #     [],    当前片文本
            #     [{...}],    后一片文本
            # ]
            for i, resp in enumerate(es_resp):
                pos = i % 3
                # 去掉无值的检索结果
                if not resp:
                    continue
                rec = resp[0]
                # 对召回的片段去重
                ans_md5 = generate_md5(rec['answer'][:100])
                if ans_md5 in dup_set:
                    continue
                # 前一片文本
                if pos == 0:
                    graph_id += 1
                    count += 1
                    recalling_records[count] = rec
                # 当前片文本
                elif pos == 1:
                    current_rec = rec['answer']
                    # 对原图表数据的answer重新赋值，添加同切片索引的文本
                    # 其中graph_ids存储了图表数据在recalling_records字典中对应的键
                    # graph_id是当前上下文数据（共三条）对应的graph_ids的索引，用于找回当前数据对应的原数据
                    recalling_records[graph_ids[graph_id]]['answer'] = f'{current_rec}\n\n{recalling_records[graph_ids[graph_id]]["answer"]}'
                # 后一片文本
                else:
                    count += 1
                    recalling_records[count] = rec
                dup_set.add(ans_md5)
            return list(recalling_records.values())


        def generate():
            try:
                milvus_future = read_thread_pool.submit(vecDataEngine.search_data, question, domain, threshold, topn, csid, search_field=search_field, output_fields=output_fields)
                es_query = {"match": {"answer": question}}
                es_future = read_thread_pool.submit(ES_STORAGE.search_documents, domain, es_query, size=topn, output_fields=output_fields)
                milvus_response = milvus_future.result()
                es_response = es_future.result()
                # 对检索结果进行去重
                #resp = milvus_response + es_response
                resp = cost_time(cross_sort)(milvus_response, es_response)
                dupli = set()
                response = dict()
                for i, record in enumerate(resp):
                    # 由于库中收录了文本长切片和该长文本的子切片，需防止前面部分重复
                    answer = record['answer'][:100]
                    a_md5 = generate_md5(answer)
                    if a_md5 in dupli:
                        continue
                    response[i] = record
                    dupli.add(a_md5)
                # 并行召回图表片段的上下文，以防信息丢失
                response = cost_time(parallel_recall)(response, dupli)

                if GLOBAL_CONFIG.rerank_config['use_xinfer'] or use_rerank:
                    logging.info('starting rerank...')
                    # 对检索结果进行重排
                    documents = []
                    for rec in response:
                        '''rec对象是一个查询结果的字典, 一般形如:
                        {
                            'question': 问题,
                            'answer': 文本,
                            'url': 对象存储路径,
                            ...
                        }
                        '''
                        q = rec.get('question', '')
                        a = rec.get('answer', '')
                        doc = f'{q}\n\n{a}'
                        documents.append(doc)
                    future = rerank_pool.submit(send_request_rerank_server, GLOBAL_CONFIG.rerank_config['url'], question, documents, topn)
                    indices, scores = future.result()
                    documents = []
                    for i, idx in enumerate(indices):
                        rec = response[idx]
                        rec.update({'score': scores[i]})
                        documents.append(rec)
                    # 统一返回结果中键的顺序
                    key_order = ['score'] + output_fields
                    response = [dict(zip(key_order, (d[key] for key in key_order))) for d in documents]
                return construct_response_result(response, csid)
            except Exception as err:
                logging.error(traceback.format_exc())
                raise err

        def es_id_to_agg(doc_ids: List[int], output_fields:List[str]):
            """通过原子文本块的唯一标识doc_id，
            查询出其对应的聚合文本块，并保证相关度排序的一致性。
            """
            order = {doc_id: i for i, doc_id in enumerate(doc_ids)}
            query = {'terms': {'doc_id': doc_ids}}
            res = ES_STORAGE.search_documents(domain, query, output_fields, size=0)
            order_rec = {}
            for rec in res:
                # order_rec = {序号: 数据字典}
                order_rec[order[rec['doc_id']]] = rec
            res = [order_rec[key] for key in sorted(order_rec)]
            return res

        def es_agg_to_text(agg_files: List[Tuple[int, str, str]], output_fields:List[str]):
            """通过原子文本块的agg_index, file_name字段，查询出对应的聚合文本块的文本。
            此外，通过原子文本块的url字段，剔除未涉及的图表markdown文本，
            只保留涉及到的图表markdown文本，以节约增强、生成环节token开销。
            """
            file_count = defaultdict(lambda: 0)
            # 存储文件对应的应保留图表
            links = defaultdict(set)
            # 构造ES查询
            should_list = []
            # 存储聚合片id及文件名构成的二元索引对应的排序
            order = dict()
            
            # 构造单个ES查询
            fill_af = lambda agg_index, file_name: {
                'bool': {
                    'must': [
                        {'term': {'agg_index': agg_index}},
                        {'term': {'file_name': file_name}},
                        {'term': {'chunk_type': 'agg_text'}}
                    ]
                }
            }
            for i, (agg_index, file_name, url) in enumerate(agg_files):
                if (agg_index, file_name) not in order:
                    order[(agg_index, file_name)] = i
                if url:
                    # 如果url不为空，则将其放入file_name键对应的列表值中
                    links[file_name].add(url)
                should_list.append(fill_af(agg_index, file_name))
            query = {'bool': {'should': should_list}}
            res = ES_STORAGE.search_documents(domain, query, output_fields, size=0)
            order_rec = dict()
            for rec in res:
                # TODO: ES中不保存空字段，为保证输出字段一定存在而做的权宜之计
                for field in output_fields:
                    if field not in rec:
                        rec[field] = ''
                # 剔除未命中原子文本块的其它图表，只保留命中图表
                file_urls = links.get(rec['file_name'], None)
                if file_urls:
                    # 字段url形如images/page19_table0.jpg
                    # 聚合长片中的url形如<table mds/page19_table0.md>
                    # 过滤时需进行替换
                    urls = [ele.replace('images', 'mds').replace('.jpg', '.md') for ele in file_urls]
                    urls = '|'.join(urls)
                    neg_pattern = r'<table (?!' + urls + r')[^>]*>[\s\S]*?</table>'
                    rec['answer'] = re.sub(neg_pattern, '', rec['answer'])
                    rec['answer'] = re.sub(r'<table .+?>', '', rec['answer']).replace('</table>', '')
                else:
                    table_pattern = r'<table .+?>[\s\S]*?</table>'
                    rec['answer'] = re.sub(table_pattern, '', rec['answer'])
                # 按文件名和相关性排序。每个文件下保留至多8片召回文本
                file_count[rec['file_name']] += 1
                if file_count[rec['file_name']] > 8:
                    continue
                order_rec[order[(rec['agg_index'], rec['file_name'])]] = rec
            final_res = [order_rec[key] for key in sorted(order_rec)]
            return final_res

        def generate_v2(output_fields:list):
            """v2 semantic search
            """
            # 需要用file_name和agg_index映射数据的顺序，确保这两个字段一定在输出字段中
            must_has = {'agg_index', 'file_name'}
            output_fields = list(set(output_fields).union(must_has))
            try:
                # 获取Milvus和ElasticSearch各自的相似原子文本块，返回其唯一标识doc_id
                milvus_future = read_thread_pool.submit(vecDataEngine.search_data, question, domain, threshold, topn, csid, search_field='vec', output_fields=['doc_id'], version=2)
                # 构造ES查询，查找chunk_type为原子切片中匹配的片段
                es_query = {
                    "bool": {
                        "must": [
                            {"term": {"chunk_type": "atom_text"}},
                            {"match": {"answer": question}}
                        ]
                    }
                }
                es_future = read_thread_pool.submit(ES_STORAGE.search_documents, domain, es_query, size=topn, output_fields=['doc_id'])
                milvus_response = milvus_future.result()
                es_response = es_future.result()
                # 对es与milvus检索结果按相关性进行穿插排序
                resp = cost_time(cross_sort)(milvus_response, es_response, version=2)
                # 使用doc_id查询原子文本块对应的聚合长篇文本块
                id_res = es_id_to_agg(resp, ['doc_id', 'agg_index', 'file_name', 'url'])
                agg_files = [(rec['agg_index'], rec['file_name'], rec.get('url', None)) for rec in id_res]
                agg_res = es_agg_to_text(agg_files, output_fields)
                # 
                return construct_response_result(agg_res, csid)
            except Exception as err:
                logging.error(traceback.format_exc())
                raise err
            
        if version == 1:
            return Response(cost_time(generate)(), mimetype="application/json")
        if version == 2:
            return Response(cost_time(generate_v2)(output_fields), mimetype="application/json")
        
    except Exception as err:
        logging.error("Faild to predict model.")
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")
    

# 数据库精准查询接口
@vecdata_blue.route("/v1/model/DataPreciseSearchEngine", methods=["POST"])
def DataPreciseSearchServer():
    """数据精确检索服务"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        question, domain, output_fields, search_field, topn, csid   = VecDataApiCommon.parse_data_precise_search_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))
        
        def generate():
            try:
                if isinstance(search_field, str):
                    query = {"term": {search_field: question}}
                    future = read_thread_pool.submit(ES_STORAGE.search_documents, domain, query, output_fields, topn)
                # TODO: 为了查询出目录语块和总结语块而做的权宜之计，日后应走elasticsearch
                elif isinstance(search_field, list):
                    future = read_thread_pool.submit(vecDataEngine.scaler_query_data, question, domain, search_field, topn, csid, output_fields=output_fields)
                response = future.result()
                # 对过长的目录做截断
                new_res = []
                for res in response:
                    if res.get('type', '') == 'outline':
                        ans = res['answer'][:525]
                        ans = ans.split('#')
                        res['answer'] = '#'.join(ans[:-1])
                    new_res.append(res)
                response = new_res
                return construct_response_result(response, csid)
            except Exception as err:
                logging.error(traceback.format_exc())
                return err.message
            
        return Response(cost_time(generate)(), mimetype="application/json")
        
    except Exception as err:
        logging.error("Faild to predict model.")
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")
    

# 查看数据集中特定字段的去重后的所有值接口
@vecdata_blue.route("/v1/model/FieldSearchEngine", methods=["POST"])
def FieldSearchServer():
    """查看数据集中特定字段的去重后的所有值接口"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        domain, output_fields, csid = VecDataApiCommon.parse_field_search_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))
        
        def generate():
            try:
                future = read_thread_pool.submit(vecDataEngine.query_field_value, domain, output_fields, csid)
                response = future.result()
                return construct_response_result(response, csid)
            except Exception as err:
                logging.error(traceback.format_exc())
                return err.message
            
        return Response(cost_time(generate)(), mimetype="application/json")
        
    except Exception as err:
        logging.error("Faild to predict model.")
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")


# 查看数据集中所有文件接口
@vecdata_blue.route("/v1/model/FileNameSearchEngine", methods=["POST"])
def FileNameSearchServer():
    """从于从对象存储中查寻指定数据集中的出所有文件名"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        domain, csid = VecDataApiCommon.parse_file_name_search_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logging.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))
        
        def generate():
            try:
                # response = vecDataEngine.query_file_names(domain, csid)
                response = ES_STORAGE.search_unique(domain, 'file_name')
                return construct_response_result(response, csid)
            except Exception as err:
                logging.error(traceback.format_exc())
                return err.message
            
        return Response(cost_time(generate)(), mimetype="application/json")
        
    except Exception as err:
        logging.error("Faild to predict model.")
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")
    

# 心跳机制接口
@vecdata_blue.route('/health', methods=['GET'])
def health():
    result = {'status': 'UP'}
    return Response(json.dumps(result), mimetype='application/json')

import base64
from module.storage.minio_storage import MINIO_STORAGE


# OCR模块上传文件到对象存储及下载
@vecdata_blue.route("/v1/model/ObjStorageEngine", methods=["POST"])
def ObjStorageServer():
    """上传文件到对象存储"""
    try:
        # 打印服务接收请求参数的耗时
        req_start_time = time.time()
        bucket_name, obj_path, obj_base64, csid, method = VecDataApiCommon.parse_obj_storage_request()
        req_end_time = time.time()
        logging.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))

        # 开始进行模型处理
        t = threading.currentThread()
        logging.info("Threading id: {}, name: {}".format(t.ident, t.getName()))

        def generate():
            try:
                if method == 'put':
                    data_bytes = io.BytesIO(base64.b64decode(obj_base64.encode('utf-8')))
                    response = MINIO_STORAGE.put_object(bucket_name=bucket_name, object_name=obj_path, data=data_bytes)
                elif method == 'get':
                    data_bytes = MINIO_STORAGE.get_object(bucket_name=bucket_name, object_name=obj_path, return_json=False)
                    response = base64.b64encode(data_bytes).decode("utf-8")
                else:
                    raise ValueError(f'Invalid method: {method}')
                return construct_response_result(response, csid)
            except Exception as err:
                logging.error(traceback.format_exc())
                return err.message
            
        return Response(cost_time(generate)(), mimetype="application/json")

    except Exception as err:
        logging.error("Faild to put obj-storage.")
        logging.error(traceback.format_exc())
        return Response(err.message, mimetype="application/json")
