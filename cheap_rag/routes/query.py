import os
from fastapi import APIRouter, HTTPException


router = APIRouter()


# local module
from utils.logger import logger
from utils.helpers import SnowflakeIDGenerator


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
        logger.info("The request data reception is completed, cost: {} s".format(req_end_time - req_start_time))
        logger.info("The current program fingerprint are: {}".format(csid))

        # 开始进行模型处理
        t = threading.currentThread()
        logger.info("Threading id: {}, name: {}".format(t.ident, t.getName()))

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
                logger.error(traceback.format_exc())
                raise err