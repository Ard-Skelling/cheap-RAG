# -*- coding: utf-8 -*-
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))


import json
import os
import re
import time
import base64
import logging
import traceback
import requests
import copy
import numpy as np
from typing import List
from datetime import datetime
from module.parsers.postprocess import summry
from module.parsers.postprocess import Inference
from pathlib import Path
from predict import BaseInferenceEngine
from configs.vector_database_config import GLOBAL_CONFIG
from concurrent.futures import ThreadPoolExecutor
from module.parsers.predict import ParseFile
from module.vecdatas.predict import PyMilvusInference
from module.spliters.predict import SplitterInference
from module.storage.minio_storage import MINIO_CONFIG, MINIO_STORAGE
from module.storage.elastic_storage import ES_STORAGE
# from module.embeddings.predict import TextEmbeddingInference
from common import cost_time, response_func, BadRequest, send_request_server, send_request_embed_server, generate_md5
from module.pipeline_v2.data_cls import DataPoint, EsDataV2, MilvusDataV2



class VecDataInference(BaseInferenceEngine):
    """"""

    def __init__(self, config):
        self.config = config
        # pymilvus 配置
        self.vecdata_config = self.config.vecdata_config
        # 大模型配置
        self.llm_config = self.config.llm_config
        # 向量模型配置
        self.embed_config = self.config.embed_config

        self.thread_pool = ThreadPoolExecutor(max_workers=self.embed_config.get("thread_num", 16))
        self.llm_thread_pool = ThreadPoolExecutor(max_workers=200)
        self.embed_thread_pool = ThreadPoolExecutor(max_workers=self.embed_config.get("thread_num", 16))   # 调用向量模型
        self.query_thread_pool = ThreadPoolExecutor(max_workers=8)

        # 初始化文本切分器对象
        self._init_spliter_infer()
        # 初始化pymilvus向量数据库对象
        self._init_pymilvus_infer()
        # 初始化文本解析器对象
        self._init_parse_infer()

    def _init_spliter_infer(self):
        """初始化文本切分器对象"""        
        spliter_config = self.config.spliter_config
        self.spliter_infer = SplitterInference(spliter_config)
        logging.info("Init spliter infer successfully.")

    def _init_pymilvus_infer(self):
        """初始化向量数据库pymilvus所需要参数"""

        self.vecdata_infer = PyMilvusInference(self.vecdata_config)
        logging.info("Init vecdata infer successfully.")
    
    @cost_time
    def _init_parse_infer(self):
        """初始化文本解析器对象"""
        # parser_config = self.config.parser_config
        self.parser_infer = ParseFile()
        logging.info("Init parser infer successfully.")

    @cost_time
    def get_vecdata_result(self, csid):
        """获取向量数据库中collection的数量"""
        try:
            return self.vecdata_infer._get_collections
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="Failed to obtain the number of vector libraries."))

    @cost_time
    def create_vecdata(self, collections, csid, version=1):
        """创建collection"""
        try:
            # 创建 elastic search 索引
            for c in collections:
                ES_STORAGE.create_index(c, version=version)
            return self.vecdata_infer._create_collection(collections, version=version)
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="create failed"))
    
    @cost_time
    def delete_vecdata(self, collections, csid):
        """删除collection"""
        try:
            for collection_name in collections:
                # 删除milvus数据集
                self.vecdata_infer._drop_collection(collection_name)
                # 删除minio存储桶-file-meta
                MINIO_STORAGE.remove_object(MINIO_CONFIG['bucket'],collection_name)
                # 删除minio存储桶-file-ocr
                MINIO_STORAGE.remove_object(MINIO_CONFIG['bucket_ocr'],collection_name)
                # 删除elasticsearch索引
                ES_STORAGE.delete_index(collection_name)
            return "delete success."
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="delete fialed"))
    
    @cost_time
    def construct_milvus_data(self, result:List[EsDataV2], domain, create_time, csid, version=1):
        try:
            if version == 1:
                question_list = []
                for i, res in enumerate(result):
                    if i == 0:
                        text = res.get('answer', '')
                    else:
                        text = res['question'] if res.get('question') else res.get('answer', '')
                    question_list.append(text)
                # 调用向量模型生成构造向量, 有question则用question字段值, 没有则用answer字段值
                vecs = cost_time(send_request_embed_server)(self.embed_config.get("url"), question_list, csid)
                for vec, res in zip(vecs, result):
                    np_vec = np.array(vec)
                    res.update({
                        "vec": np_vec,
                        "slice_vec": np.zeros_like(vec),
                        "q_slice_vec": np.zeros_like(vec),
                        "collection_name": domain,
                        "create_time": create_time
                    })
                    # 由于milvus数据库的bug，会将·识别为不止1个字符，导致文本长度超限，需要特殊处理
                    # 被截断的文本已在divide_results方法中进行处理，不会丢失
                    res["answer"] = re.sub('·+', '··', res["answer"])[:9000]
                return result

            elif version == 2:
                question_list = []
                for i, res in enumerate(result):
                    if i == 0:
                        text = res.answer
                    else:
                        text = res.question if res.question else res.answer
                    question_list.append(text)
                # 调用向量模型生成构造向量, 有question则用question字段值, 没有则用answer字段值
                vecs = cost_time(send_request_embed_server)(self.embed_config.get("url"), question_list, csid)
                data = []
                for vec, res in zip(vecs, result):
                    data.append(MilvusDataV2(doc_id=res.doc_id, vec=vec))
                return data
                
            else:
                raise NotImplementedError(f'Unsupported version: {version}')

        except Exception as err:
            logging.error(traceback.format_exc())
            raise "Faild to construct data."

    @staticmethod
    def chunking_large_test(text, chunk_size=1000, overlap=300):
        """
        切分长文本为多个块，每块约1000字，前后有100字的交叠。
        通过优先级标点符号（句号、逗号等）来确定交叠部分。

        :param text: 待切分的长文本
        :param chunk_size: 每块的字数，默认1000字
        :param overlap_size: 前后块之间的重叠字数，默认100字
        :return: 返回切分后的文本块列表
        """
        # 优先级标点符号，越前面的越优先
        priority_punctuation = ['。', '？', '！']
        # 清洗文本，去掉多余的空格
        text = text.strip()
        # 切分后的文本块
        chunks = []
        # 当前处理的位置
        start = 0
        offset = int(overlap / 2)
        chunk_size = chunk_size - offset
        end = chunk_size
        # 小于主片段+子片段长度直接返回
        if len(text) <= chunk_size + overlap:
            return [text]
        while start < len(text):
            overlap_end = end + overlap
            if overlap_end >= len(text):
                chunks.append(text[start:])  # 如果没有足够的字符，最后一块直接加进去
                break
            # 确定切分点（从当前块的结束位置向后查找优先级标点符号）
            cut_point = -1
            for i in range(end, overlap_end):
                if i < len(text):
                    # 如果找到优先级标点符号之一
                    if text[i] in priority_punctuation:
                        cut_point = i + 1  # 设置为切分点的后一位置
                        break
            # 如果没有找到合适的切分点，默认按字数切分
            if cut_point == -1:
                cut_point = overlap_end
            # 生成文本块
            chunks.append(text[start:cut_point])
            start = cut_point
            end = start + chunk_size
        return chunks 

    def divide_results(self, results: List[dict], version=1):
        def is_to_chunk(before:dict, after:dict):
            """检定before块是否属于需要切分的长文本语块"""
            # 回答超字数限制
            cri_0 = len(before['answer']) > GLOBAL_CONFIG.spliter_config['threshold']
            cri_1 = before['type'] == 'text'
            cri_2 = before['answer'] != after['answer']
            return cri_0 and cri_1 and cri_2
        
        def build_sub_chunks(record:dict):
            text = record['answer']
            sub_chunks = self.chunking_large_test(
                text, 
                chunk_size=GLOBAL_CONFIG.spliter_config['chunk_size'],
                overlap=GLOBAL_CONFIG.spliter_config['overlap']
            )
            sub_records = []
            for chunk in sub_chunks:
                data_point = copy.deepcopy(record)
                data_point['answer'] = chunk
                sub_records.append(data_point)
            return sub_records

        def deal_with_cell(table_md, cell_data_set:set, cell_data_list:list):
            pattern_cell = r'(?<=|)[^|]+(?=|)'
            pattern_zh_en = r'[一-龥a-zA-Z]'
            res = re.findall(pattern_cell, table_md)
            for cell in res:
                if re.search(pattern_zh_en, cell):
                    cell = cell.strip()
                    if cell not in cell_data_set:
                        cell_data_list.append(cell)
                        cell_data_set.add(cell)
            return cell_data_set, cell_data_list


        def deal_with_row_col(line, col_data_set:set, col_data_list:list, first_row:str):
            res = re.search(r'^\s*\|([^|]+)\|', line)
            if res:
                cell_content = res.group(1).strip()
                if cell_content:
                    if not first_row:
                        first_row = line.strip()
                    if cell_content not in col_data_set:
                        col_data_set.add(cell_content)
                        col_data_list.append(cell_content)
            return col_data_set, col_data_list, first_row


        def table_addon_info(table_md:str):
            table_md = table_md.strip()

            # 获取包含中英字符的去重后单元格的值
            cell_data_list = []
            cell_data_set = set()
            cell_data_set, cell_data_list = deal_with_cell(table_md, cell_data_set, cell_data_list)

            # 获取表格首行首列
            lines = table_md.split('\n')
            col_data_list = []
            col_data_set = set()
            first_row = ''
            for line in lines:
                if not line.strip():
                    continue
                col_data_set, col_data_list, first_row = deal_with_row_col(line, col_data_set, col_data_list, first_row)
            
            first_col = '|'.join(col_data_list)
            cell_str = '|'.join(cell_data_list)
            return first_row, first_col, cell_str

        def reindex(results_list:list):
            # TODO: 改为管道模式，节省内存
            current_index = 1
            new_res_list = []
            agg_block = ''
            pre_block = ''
            agg_len = 0
            atom_index = 1
            agg_index = 1
            addon_dict = dict()
            raw_indices = [rec['index'] for rec in results_list]
            done_table = set()
            offset = 0
            for i, rec in enumerate(results_list):
                index = rec['index']
                type_ = rec['type']
                # csv、excel中存在str变int的情况
                rec['question'] = str(rec['question'])
                rec['answer'] = str(rec['answer'])
                new_rec = copy.deepcopy(rec)
                if version == 2:
                    # 为每条数据增加doc_id为唯一标识
                    new_rec['doc_id'] = generate_md5(f'{new_rec["question"]}{new_rec["answer"]}')
                    # 生成atom_text片，是真正用于向量检索的文本片，
                    # 对answer字段按换行符切分
                    if rec['type'] in ['text', 'table', 'image']:
                        if rec['type'] == 'text':
                            answers = [ele.strip() for ele in rec['answer'].split('\n')]
                            for ans in answers:
                                # 每次对数据进行操作都创建新内存对象
                                new_new_rec = copy.deepcopy(rec)
                                ans_md5 = generate_md5(ans)
                                new_new_rec['answer'] = f'{rec["question"]}{ans}'
                                new_new_rec['question'] = ''
                                new_new_rec['type'] = 'atom_text'
                                new_new_rec['atom_index'] = atom_index
                                new_new_rec['agg_index'] = agg_index
                                new_new_rec['doc_id'] = ans_md5
                                addon_dict[ans_md5] = new_new_rec
                                atom_index += 1
                        # 图表存在多片相同answer，原子切片中只保留问题
                        else:
                            new_new_rec = copy.deepcopy(rec)
                            q_md5 = generate_md5(rec['question'])
                            new_new_rec['type'] = 'atom_text'
                            new_new_rec['atom_index'] = atom_index
                            new_new_rec['agg_index'] = agg_index
                            new_new_rec['doc_id'] = q_md5
                            addon_dict[q_md5] = new_new_rec
                            atom_index += 1
                        # 生成agg_text片，是真正返回的文本片，累计长度超过agg_length时切分
                        # 不处理索引为0的片段，这种片段一般是总结、大纲等全局信息
                        ans_len = len(rec['answer'])
                        if ans_len + agg_len >= GLOBAL_CONFIG.spliter_config['agg_length']:
                            new_new_rec = copy.deepcopy(rec)
                            ans_md5 = generate_md5(agg_block)
                            new_new_rec['answer'] = agg_block
                            new_new_rec['type'] = 'agg_text'
                            new_new_rec['agg_index'] = agg_index
                            new_new_rec['question'] = ''
                            new_new_rec['doc_id'] = ans_md5
                            new_new_rec['url'] = ''
                            new_new_rec['index'] = 0
                            # 当拼接长度加当前文本长度超限时，将之前的拼接文本保存为大长片
                            addon_dict[ans_md5] = new_new_rec
                            agg_index += 1
                            agg_len = 0
                            # 从当前文本继续往后拼接
                            agg_block = rec['answer']
                        else:
                            # 将不重复的answer片进行拼接
                            if not rec['answer'] == pre_block:
                                agg_len += ans_len
                                agg_block = f'{agg_block}\n\n{rec["answer"]}'
                                pre_block = rec['answer']
                if index == 0:
                    new_res_list.append(new_rec)
                    continue
                if type_ == 'text':
                    new_rec['index'] = current_index
                    current_index += 1
                    new_res_list.append(new_rec)
                    
                elif type_ in ['table', 'image']:
                    # 防止answer超长
                    new_rec['answer'] = new_rec['answer'][:5000]
                    # 防止图表片段数组越界
                    # 第一片索引为1
                    if i == 0:
                        new_rec['index'] = current_index
                    else:
                        # 若与前片索引相同，使用前片索引
                        if new_rec['index'] == raw_indices[i - 1]:
                            new_rec['index'] = new_res_list[i - 1]['index']
                        # 若与前片索引不同，使用当前新索引，为上片文本索引+1
                        else:
                            new_rec['index'] = current_index
                    # 用表格首行首列和表格中英文内容做成question
                    if type_ == 'table':
                        ans = new_rec['answer']
                        first_row, first_col, cell_str = table_addon_info(ans)
                        # 对首行和首列做截断，以防超出milvus字段字数限制
                        # 此外，embedding也无法处理512个token以上的输入
                        q_row_col = f'表格首行：{first_row[:256]}\n表格首列：{first_col[:256]}'
                        q_cell = f'表格文本：{cell_str[:512]}'
                        for q in [q_row_col, q_cell]:
                            q_md5 = generate_md5(q)
                            if q_md5 not in done_table:
                                table_rec = copy.deepcopy(new_rec)
                                table_rec['question'] = q
                                new_res_list.append(table_rec)
                                # 由于增加了元素，
                                # 需向raw_indices对应位置
                                # 插入当前扩展片段在旧索引体系下的对应索引
                                raw_indices.insert(i + offset, index)
                                offset += 1
                                done_table.add(q_md5)
                    new_res_list.append(new_rec)
                else:
                    new_res_list.append(new_rec)

            if version == 1:
                return new_res_list
            if version == 2:
                # 收录最后一片到大长片中
                if not rec['answer'] == pre_block:
                    new_new_rec = copy.deepcopy(rec)
                    agg_block = f'{agg_block}\n\n{rec["answer"]}'.strip()
                    new_new_rec['answer'] = agg_block
                    new_new_rec['type'] = 'agg_text'
                    new_new_rec['agg_index'] = agg_index
                    new_new_rec['question'] = ''
                    new_new_rec['url'] = ''
                    new_new_rec['index'] = 0
                    ans_md5 = generate_md5(rec['answer'])
                    new_new_rec['doc_id'] = ans_md5
                    addon_dict[ans_md5] = new_new_rec
                return new_res_list + list(addon_dict.values())

        new_results = []
        for i in range(1, len(results)):
            before = results[i - 1]
            after = results[i]
            if is_to_chunk(before, after):
                sub_records = build_sub_chunks(before)
                new_results.extend(sub_records)
            else:
                new_results.append(before)
        last_rec = results[-1]
        # 对最后一个切片进行校验
        if len(last_rec['answer']) > GLOBAL_CONFIG.spliter_config['threshold'] and last_rec['type'] == 'text':
            sub_records = build_sub_chunks(last_rec)
            new_results.extend(sub_records)
        else:
            new_results.append(last_rec)

        # 重排索引，并将表格的首行首列、及存在中英文本的单元格聚合构造为两个问题
        new_results = reindex(new_results)
        return new_results


    @cost_time
    def process_insert_data(self, results, domain, csid, is_document=False, version=1):
        """
            多线程处理将数据插入库中
            param results: 输出的数据条数
            param domain: 待插入的库名

        Params:
            results(List[dict]): 文本原始切片，形如：
                {
                    "question": "qqq", 
                    "answer": "aaa", 
                    "type": "text", 
                    "url": "", 
                    "parent_title": "", 
                    "title": "", 
                    "index": 0, 
                    "file_name": "xxx.pdf"
                }
            domain(str): 知识库名称
        """
        data = []
        cn = 0
        create_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")  # 记录每一条的创建时间
        future_list = []
        temp_result = []
        file_name = results[0]['file_name']
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            # 删除milvus数据
            futures.append(executor.submit(self.vecdata_infer._delete_data, [{"file_name": file_name}], domain))
            # 删除elasticsearch数据
            futures.append(executor.submit(ES_STORAGE.delete_doc(domain, query={"terms": {"file_name": [file_name]}})))

        # 对长文本的answer做进一步的切分，chunk_size由config中的参数来控制，形成三级文本块
        results = self.divide_results(results, version=version)

        if version == 1:
            for result in results:
                # 这里不要查直接删(耗时, 用时间标记)
                # self.vecdata_infer._delete_data([{"file_name": result["file_name"]}], domain)
                if not temp_result:
                    temp_result.append(result)
                else:
                    if result["answer"] == temp_result[-1]["answer"]:
                        temp_result.append(result)
                    else:
                        cn += len(temp_result)
                        future = self.embed_thread_pool.submit(self.construct_milvus_data, temp_result, domain, create_time, csid, version=version)
                        future_list.append(future)
                        temp_result = [result]
            if temp_result:
                cn += len(temp_result)
                future = self.embed_thread_pool.submit(self.construct_milvus_data, temp_result, domain, create_time, csid, version=version)
                future_list.append(future)

            for future in future_list:
                _result = future.result()
                data.extend(_result)

            # 写入es
            self.insert_es(data, domain, version)
            # 写入milvus
            self.vecdata_infer._insert_data(data, domain, version=version)   

        # 区分需要生成embedding和无需生成的数据
        # v2 版本启用了pydantic.BaseModel来做数据类，进行字段清洗和数据类型校验
        elif version == 2:
            no_ems = []
            to_ems = []
            for rec in results:
                if rec['type'] == 'atom_text':
                    to_ems.append(EsDataV2(create_time=create_time, **rec))
                else:
                    no_ems.append(EsDataV2(create_time=create_time, **rec))
            # 生成embedding
            batch_size = GLOBAL_CONFIG.embed_config['batch_size']
            for i in range(0, len(to_ems), batch_size):
                future = self.embed_thread_pool.submit(self.construct_milvus_data, to_ems[i:i + batch_size], domain, create_time, csid, version=version)
                future_list.append(future)
            for future in future_list:
                _result = future.result()
                data.extend(_result)
            # 分别写入不同的数据库
            self.insert_es(no_ems + to_ems, domain, version)
            self.vecdata_infer._insert_data(data, domain, version=version)

        else:
            raise NotImplementedError(f'Unsupported version: {version}')

    def insert_es(self, data:List[DataPoint], domain:str, version=1):
        if version == 1:
            results = []
            for rec in data:
                es_data = dict()
                for k, v in rec.items():
                    if k in ['vec', 'q_slice_vec', 'slice_vec', 'collection_name']:
                        continue
                    elif k == 'type':
                        es_data['chunk_type'] = v
                    else:
                        es_data[k] = v
                    if k == 'answer':
                        q = rec.get('question', '')
                        es_data['_id'] = generate_md5(f'{q}{v}')
                results.append(es_data)
            ES_STORAGE.bulk_insert_documents(domain, results)
        if version == 2:
            results = [d.model_dump() for d in data]
            ES_STORAGE.bulk_insert_documents(domain, results)

    @cost_time
    def insert_data(self, document, domain, file_name, file_meta, csid, batch=1000, is_enhanced=False, version=1):
        """插入数据"""
        try:
            # 先检验数据集是否存在
            collection = self.vecdata_infer._check_collection_state(domain)
            if not collection:
                logging.error("Faild to load collection.")
                raise ValueError(f'Faild to load collection: {domain}') 
            future_list = []
            bytes_data = base64.b64decode(document.encode("utf-8"))
            _, ext = os.path.splitext(file_name)
            file_type = ext.lower()
            if file_type in [".xlsx", ".xls", ".csv", ".json", ".txt"]:
                results = []
                pub_meta = {
                    "index": 0,
                    "type": "qa",
                    "collection_name": domain,
                    "parent_title": "",
                    "title": "",
                    "url": ""
                }
                if file_type in [".json"]:
                    for values in self.parser_infer.get_json_iters(bytes_data):
                        pri_data = {"question": values["question"], "answer": values["answer"], "file_name": file_name}
                        pri_data.update(pub_meta)
                        results.append(pri_data)
                        if len(results) == batch:
                            future = self.thread_pool.submit(self.process_insert_data, results, domain, csid, version=version)
                            future_list.append(future)
                            results = []
                    
                elif file_type in [".xlsx", ".xls", ".csv"]:
                    for values in self.parser_infer.get_xlsx_result(bytes_data, suffix=file_type):
                        pri_data = {"question": values["question"], "answer": values["answer"], "file_name": file_name}
                        pri_data.update(pub_meta)
                        results.append(pri_data)
                        if len(results) == batch:
                            future = self.thread_pool.submit(self.process_insert_data, results, domain, csid, version=version)
                            future_list.append(future)
                            results = []

                elif file_type in [".txt"]:
                    bytes_data = self.parser_infer.check_encoding(bytes_data)
                    content = bytes_data.decode()
                    # 获取文本总结语块，在send_request_server中按最长25000字截断
                    zj = summry.format(content)
                    summary = send_request_server(self.llm_config["url"],zj)
                    summary_dict = json.loads(summary)
                    pri_data = {"question": "", "answer": summary_dict['choices'][0]['message']['content'], "file_name": file_name}
                    pri_data.update(pub_meta)
                    pri_data.update({'type': 'summary'})
                    results.append(pri_data)
                    for values in self.parser_infer.get_txt_result(bytes_data):
                        pri_data = {"question": values["question"], "answer": values["answer"], "file_name": file_name}
                        pri_data.update(pub_meta)
                        results.append(pri_data)
                        if len(results) == batch:
                            future = self.thread_pool.submit(self.process_insert_data, results, domain, csid, version=version)
                            future_list.append(future)
                            results = []
                
                else:
                    raise NotImplementedError
                
                # 不够一个批次
                if results:
                    future = self.thread_pool.submit(self.process_insert_data, results, domain, csid, version=version)
                    future_list.append(future)
                    results = []
            else:
                
                start_time = time.time()
                results = self.parser_infer.get_document_result(bytes_data,domain, file_name, csid, is_enhanced)

                # import json
                # with open("./ocrs/GB 14048.1-2012 低压开关设备和控制设备 第1部分 总则.pdf.json", "r", encoding="utf-8") as f:
                #     results = json.load(f)

                logging.info(f"------------results------------{len(results)}")
                logging.info("ocr and llm extract question cost: {} s".format(time.time() - start_time))
                # for i in range(len(results), 1000):
                    # future = self.thread_pool.submit(self.process_insert_data, results, domain, csid, is_document=True)
                future = self.thread_pool.submit(self.process_insert_data, results, domain, csid, is_document=True, version=version)
                future_list.append(future)
           
            for future in future_list:
                _ = future.result()

            # 将文件元数据写入对象存储，用于检索数据集中的全量文件名，对象在minio中的名称形如：collection_name/demo_doc.doc.json
            if file_meta is None:
                file_meta = dict()
            MINIO_STORAGE.put_object(f'{domain}/{file_name}.json', data=file_meta)
            return "insert success"
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="insert failed"))
    
    @cost_time
    def delete_data(self, domain, file_name, csid):
        """删除数据"""
        def delete_milvus(domain, file_name):
            if isinstance(file_name, str):
                filter_ = [{"file_name": file_name}]
            else:
                filter_ = [{'file_name': f} for f in file_name]
            self.vecdata_infer._delete_data(filter_, domain)

        def delete_minio(domain, file_name):
            # 删除file-meta桶中的文件元数据
            file_name = file_name if isinstance(file_name, list) else [file_name]
            del_list = [f'{domain}/{f}.json' for f in file_name]
            errs = MINIO_STORAGE.remove_object(MINIO_CONFIG['bucket'],object_list=del_list)
            if errs:
                for e in errs:
                    logging.error(e)
                # TODO: 由于版本未实现完全升级, 一些数据集新旧存储方式混用，不能在这里升起对象存储的删除报错，只能暂时忽略
                # raise IOError(f'Delete Minio failed: {file_name}')

            # 删除file-ocr桶中的ocr结果文件
            errs_ocr = MINIO_STORAGE.remove_object(MINIO_CONFIG['bucket_ocr'],prefix=f'{domain}/{file_name}')
            if errs_ocr:
                for e in errs_ocr:
                    logging.error(e)

        def delete_es(domain, file_name):
            file_name = file_name if isinstance(file_name, list) else [file_name]
            ES_STORAGE.delete_doc(domain, query={"terms": {"file_name": file_name}})

        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                # 删除milvus数据
                futures.append(executor.submit(delete_milvus, domain, file_name))
                # 删出minio数据
                futures.append(executor.submit(delete_minio, domain, file_name))
                # 删除elasticsearch数据
                futures.append(executor.submit(delete_es, domain, file_name))
            return "delete success"
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="delete failed"))
    
    @cost_time
    def search_data(self, question, domain, threshold, topn, csid, search_field="vec", output_fields=[], version=1):
        """查找数据"""
        try:
            final_result = []
            logging.info("collection_name: {}".format(domain))
            # 用户输入向量化
            vecs = cost_time(send_request_embed_server)(self.embed_config.get("url"), question, csid)
            query_vec = np.array(vecs[0])
            # 向量库检索
            search_data_list = cost_time(self.vecdata_infer._search_data)([query_vec], domain, search_field=search_field, output_fields=output_fields, topn=topn)
            if version == 2:
                return search_data_list
            # 根据得分进行排序
            search_data_list.sort(key=lambda x: x["score"])
            
            dup_answer = []
            dup_search_data_list = []
            for data in search_data_list[::-1]:
                if not dup_answer or data["answer"] not in dup_answer:
                    dup_answer.append(data["answer"])
                    dup_search_data_list.append(data)

            # for data in search_data_list[::-1][:topn]:
            for data in dup_search_data_list[:topn]:
                if data["score"] > threshold:
                    output_field_dict = {}
                    output_field_dict["score"] = data["score"]
                    for field in output_fields:
                        output_field_dict[field] = data.get(field, "")
                    final_result.append(output_field_dict)
            logging.debug("final_result: {}".format(final_result))
            return final_result
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="search failed"))
    
    @cost_time
    def precise_search_data(self, question, domain, search_field, topn, csid, output_fields=[]):
        """根据指定字段精确查找数据"""
        try:
            final_result = []
            logging.info("collection_name: {}".format(domain))
            search_data_list = cost_time(self.vecdata_infer._query_data)(question, domain, primary_key=search_field, topn=topn)
            
            for data in search_data_list:
                output_field_dict = {}
                for field in output_fields:
                    output_field_dict[field] = data.get(field, "") 
                final_result.append(output_field_dict)
            logging.info("final_result: {}".format(final_result))
            return final_result
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="precise search failed"))

    @cost_time
    def scaler_query_data(self, question, domain, search_field, topn, csid, output_fields=None, operator='=='):
        """根据指定字段精确查找数据"""
        try:
            final_result = []
            logging.info("collection_name: {}".format(domain))
            expr = []
            if not isinstance(question, list):
                question = [question]
                search_field = [search_field]
                operator = [operator]
            if operator == ['=='] or operator == '==':
                operator = ['==' for _ in question]
            for field, op, value in zip(search_field, operator, question):
                if isinstance(value, str):
                    value = f'"{value}"'
                expr.append(f'{field} {op} {value}')
            expr = ' && '.join(expr)
            search_data_list = cost_time(self.vecdata_infer._scalar_query_data)(
                domain,
                expr,
                output_fields,
                end_limit=topn
            )
            for data in search_data_list:
                output_field_dict = {}
                for field in output_fields:
                    output_field_dict[field] = data.get(field, "") 
                final_result.append(output_field_dict)
            logging.info("final_result: {}".format(final_result))
            return final_result
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="precise search failed"))

    @cost_time
    def query_field_value(
        self, 
        domain:str,
        output_fields:List[str],
        csid:str,
    ):      
        # 获取指定字段的值的集合，以List[Dict[str, List[str]]]形式返回
        try:
            logging.info("collection_name: {}".format(domain))
            result = cost_time(self.vecdata_infer._query_field_value)(domain, output_fields, self.query_thread_pool)
            return result
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="precise search failed"))

        
    @cost_time
    def query_file_names(
        self, 
        domain:str,
        csid:str,
    ):      
        # 从对象存储获取数据集中所有文件的文件名
        # 使用对象存储的原因是milvus当前版本不支持全量查询，最多只能返回16384条数据，无法穷尽所有数据
        try:
            logging.info("collection_name: {}".format(domain))
            result = [Path(file_name).stem for file_name in MINIO_STORAGE.list_objects(
                bucket_name=GLOBAL_CONFIG.minio_config['bucket'], 
                prefix=f'{domain}/')]
            return result
        except Exception as err:
            logging.error(traceback.format_exc())
            raise BadRequest(response_func(csid=csid, code="10003", message="precise search failed"))
        
