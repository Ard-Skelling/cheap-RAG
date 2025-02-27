import re
import asyncio
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict


# local module
from common.utils import generate_md5
from configs.config_v2.config_cls import (
    ChunkingConfig,
    InsertPreprocessingConfig
)
from configs.config_v2.config import (
    CHUNKING_CONFIG,
    INSERT_PRE_CONFIG
)
from module.pipeline_v2.data_cls import (
    Task,
    AggChunk,
    MilvusDataV2,
    GraphInfo,
    TableChunk,
    AtomChunk
)
from module.pipeline_v2.tool_calling import LlmApi, EmbeddingApi


class Chunking:
    def __init__(self, pool: ProcessPoolExecutor, task:Task, config: ChunkingConfig = None) -> None:
        self.config = config or CHUNKING_CONFIG
        self.pool = pool
        # 尽量只从self.task中读值，如果要修改，要加进程锁
        self.task = task
        self.init_chunking_params()
        self.init_result_dicts()


    def init_result_dicts(self):
        self.agg_chunks = defaultdict(list)
        # 不解析版面结构的话，存这玩意儿意义不大
        # self.standard_chunks = dict()
        self.atom_chunks = []

    def init_chunking_params(self):
        self.agg_thresold = self.config.agg_size
        self.agg_overlap = self.config.agg_overlap
        half_overlap = int(self.agg_overlap / 2)
        self.agg_enter = self.agg_thresold - half_overlap
        self.agg_exit = self.agg_thresold + half_overlap
        self.atom_thredhold = self.config.atom_size

    @staticmethod
    def has_many_dots(chunk: str):
            """用于判别目录文本块"""
            crit_0 = chunk.count('·') >= 5 or chunk.count('·') / len(chunk) >= 0.3
            crit_1 = chunk.count('.') >= 5 or chunk.count('.') / len(chunk) >= 0.3
            return crit_0 or crit_1

    @staticmethod
    def replace_nonsense(text, is_title=False):
        """处理文本中的无意义字符。Milvus会将某些字符识别为不止一个字符"""
        if is_title:
            text = re.sub('[^0-9a-zA-Z一-龥 \.·,，。]', '', text)
        text = re.sub(r'···+', '···', text)
        text = re.sub(r'。。+', '。。', text)
        text = re.sub(r'\.\.\.+', '...', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    @staticmethod
    def standardize_chunk(chunk:str):
        chunk = chunk.strip()
        # 替换句首标题标识#
        chunk = re.sub('^#+', '', chunk)
        # 处理文本中的非句末换行
        is_directory = Chunking.has_many_dots(chunk)
        if is_directory:
            # 替换目录中的"......"
            chunk = Chunking.replace_nonsense(chunk)
        else:
            # 如果不是目录，则处理非句末换行
            chunk = re.sub('(?<=[^。?!？！])\n', '', chunk)
        chunk_size = len(chunk)
        return chunk_size, chunk


    def replace_link(self, chunk:str):
        "替换文本块中的链接为指定的内容，主要是table, image, formula三种形式"
        pattern = r'!\[\]\((\w+/page\d+_(\w+)\d+\.(\w+))\)'
        match_res = re.match(pattern, chunk)
        if match_res:
            # 处理链接
            link, chunk_type, chunk_format = match_res.groups()
            file_name = Path(link).name
            if chunk_format == 'jpg':
                if chunk_type == 'table':
                    chunk = re.sub(pattern, f'表格截图链接：![]({link})', chunk)
                    return 'table', chunk, link
                elif chunk_type == 'formula':
                    chunk = re.sub(pattern, f'公式截图链接：![]({link})', chunk)
                    return 'formula', chunk, link
                else:
                    chunk = chunk = re.sub(pattern, f'图片链接：![]({link})', chunk)
                    return 'image', chunk, link
            elif chunk_format == 'md':
                if chunk_type == 'table':
                    if self.config.agg_table_md:
                        # 替换原内容为表格markdown，用<table link>...</table>进行定位
                        chunk = f'<table {link}>{self.task.result[file_name]}</table>'
                        return 'table_md', chunk, link
                    else:
                        # 替换原内容为空
                        return 'table_md', '', link
                if chunk_type == 'image':
                    # 替换原内容为图片OCR结果markdown，用<image link>...</table>进行定位
                    if self.config.agg_image_md:
                        chunk = f'<image {link}>{self.task.result[file_name]}</image>'
                        return 'image_md', self.task.result[file_name], link
                    else:
                        # 替换原内容为空
                        return 'image_md', '', link
            else:
                raise NotImplementedError(f'Unsupported chunk_format: {chunk_format}')
        else:
            # 处理纯文本
            return 'text', chunk, ''

    def cumulate_agg(
        self, 
        belong_index,
        agg_index,
        agg_cumu,
        chunk
    ):
        # 替换文本片内的链接
        chunk_type, chunk, link = self.replace_link(chunk)
        chunk_size = len(chunk)
        # 将当前片追加到当前agg_index
        self.agg_chunks[agg_index].append(chunk.strip())
        agg_cumu += chunk_size
        if agg_cumu > self.agg_enter:
            # 当前片已进入overlap区间，需加入下一聚合片
            self.agg_chunks[agg_index + 1].append(chunk.strip())
            if agg_cumu > self.agg_thresold:
                # 当前片超越overlap的中间点，需更新用于原子片的belong_index变量
                belong_index = agg_index + 1
                if agg_cumu > self.agg_exit:
                    # 当前片已越过overlap，需重启用新的agg_index
                    agg_index += 1
                    agg_cumu -= self.config.agg_size
        return chunk_type, chunk, link, belong_index, agg_index, agg_cumu


    def cumulate_atom(
        self,
        atom_index,
        atom_cumu,
        temp_atom,
        belong_index,
        chunk_index,
        chunk,
        chunk_type,
        chunk_size,
        link,
        is_last
    ):
        # TODO: 如果当前chunk过长，需切分为比较短小的原子片段
        temp_dict = {
            'atom_index': atom_index,
            'index': chunk_index,
            'agg_index': belong_index,
            'answer': chunk,
            'chunk_type': 'atom_text',
            'url': ''
        }
        if chunk_type == 'text':
            atom_cumu += chunk_size
            temp_atom = f'{temp_atom}\n\n{chunk}'
            # 累积原子片长度到切分阈值
            if atom_cumu > self.atom_thredhold or is_last:
                temp_dict.update({'answer': temp_atom.strip()})
                self.atom_chunks.append(temp_dict)
                atom_cumu = 0
                temp_atom = ''
                atom_index += 1
        elif chunk_type in ['table_md', 'image_md']:
            # 将之前的累积原子片段提交self.atom_chunks
            if atom_cumu != 0:
                temp_dict.update({'answer': temp_atom.strip()})
                self.atom_chunks.append(temp_dict)
                atom_cumu = 0
                temp_atom = ''
                atom_index += 1
                temp_dict = {
                    'atom_index': atom_index,
                    'index': chunk_index,
                    'agg_index': belong_index,
                    'answer': chunk,
                    'chunk_type': 'atom_text',
                    'url': ''
                }
            # 将md后缀换为jpg
            link = str(Path(link).with_suffix('.jpg'))
            temp_dict.update({'chunk_type': chunk_type, 'url': link})
            self.atom_chunks.append(temp_dict)
            atom_index += 1
        else:
            # table, image 和 formula 的链接对原子化语义和文本召回无关紧要
            # 会与上下文一起在agg_text中返回
            pass
        return atom_index, atom_cumu, temp_atom


    def chunking(self, doc: str):
        agg_index = 1    # 聚合片索引，从1开始
        agg_cumu = 0    # 截止当前累积的聚合片长度，用于聚合长文本片
        belong_index = 1    # atom文本块从属的agg块的index
        atom_index = 1
        atom_cumu = 0    # 截止当前累积的原子片长度，用于聚合原子文本片
        temp_atom = ''

        # TODO: 如果需对文本块内对象进行计算密集型任务，可先划分任务批次
        # 再使用self.pool来执行
        ...

        # 以下是chunking的主逻辑:
        chunks = doc.split('\n\n')
        chunks_len = len(chunks)
        for i, block in enumerate(chunks):
            is_last = i == chunks_len - 1
            chunk_size, chunk = self.standardize_chunk(block)
            # self.standard_chunks[i] = chunk
            # TODO: 处理嵌套的版面

            # 处理聚合片的划分，顺便鉴定片段类型
            chunk_type, chunk, link, belong_index, agg_index, agg_cumu = self.cumulate_agg(
                belong_index=belong_index,
                agg_index=agg_index, 
                agg_cumu=agg_cumu, 
                chunk=chunk
            )

            # 拼接和做成原子片
            atom_index, atom_cumu, temp_atom = self.cumulate_atom(
                atom_index=atom_index, 
                atom_cumu=atom_cumu, 
                temp_atom=temp_atom, 
                belong_index=belong_index,
                chunk_index=i,
                chunk=chunk, 
                chunk_type=chunk_type,
                chunk_size=chunk_size,
                link=link,
                is_last=is_last
            )

        result = {
            'agg_chunks': {k: '\n\n'.join(v) for k, v in self.agg_chunks.items()},
            'atom_chunks': self.atom_chunks
        }
        return result


class InsertPreprocessing:
    def __init__(self, pool:ProcessPoolExecutor, llm:LlmApi, embedding:EmbeddingApi, config:InsertPreprocessingConfig = None) -> None:
        self.config = config or INSERT_PRE_CONFIG
        self.pool = pool
        self.llm = llm
        self.embedding = embedding

    @staticmethod
    def get_agg_chunk(domain, file_name, agg_index, agg_chunk):
        return AggChunk(
            domain=domain,
            file_name=file_name,
            doc_id=generate_md5(agg_chunk),
            answer=agg_chunk,
            agg_index=agg_index
        ).model_dump()
    

    async def process_agg(self, domain:str, file_name:str, agg_chunks:dict):
        # loop = asyncio.get_running_loop()
        # aggs = [loop.run_in_executor(self.pool, self.get_agg_chunk, \
        #     domain, file_name, agg_index, agg_chunk) \
        #     for agg_index, agg_chunk in agg_chunks.items()]
        # aggs = await asyncio.gather(*aggs)
        aggs = [self.get_agg_chunk(domain, file_name, agg_index, agg_chunk) \
            for agg_index, agg_chunk in agg_chunks.items()]
        # 返回待写入es的agg_chunk数据
        logging.info(f'agg_text processed: {file_name}, count: {len(aggs)}')
        return aggs

    
    async def to_embedding(self, file_name, emb_queue:asyncio.Queue, batch_size:int):
        emb_results = dict()
        doc_ids = []
        docs = []
        count = 0
        while True:
            await asyncio.sleep(0.1)
            doc = await emb_queue.get()
            if doc is None:
                emb_queue.task_done()
                break
            doc_id, doc_text = doc
            doc_ids.append(doc_id)
            docs.append(doc_text)
            count += 1
            if count == batch_size:
                res = await self.embedding.send_embedding(docs)
                res = dict(zip(doc_ids, res))
                emb_results.update(res)
                doc_ids = []
                docs = []
                count = 0
            emb_queue.task_done()
        # 处理残余数据
        if doc_ids:
            res = await self.embedding.send_embedding(docs)
            res = dict(zip(doc_ids, res))
            emb_results.update(res)
        emb_results = [MilvusDataV2(doc_id=doc_id, vec=vec, file_name=file_name) for doc_id, vec in emb_results.items()]
        logging.info(f'Embedding finished: {file_name}, count: {len(emb_results)}')
        return emb_results
    

    @staticmethod
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


    @staticmethod
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


    async def table_addon_info(self, emb_queue:asyncio.Queue, table_md:str):
        table_result = dict()

        table_md = table_md.strip()
        # 获取包含中英字符的去重后单元格的值
        cell_data_list = []
        cell_data_set = set()
        cell_data_set, cell_data_list = self.deal_with_cell(table_md, cell_data_set, cell_data_list)

        # 获取表格首行首列
        lines = table_md.split('\n')
        col_data_list = []
        col_data_set = set()
        first_row = ''
        for line in lines:
            if not line.strip():
                continue
            col_data_set, col_data_list, first_row = self.deal_with_row_col(line, col_data_set, col_data_list, first_row)
        
        first_col = '|'.join(col_data_list)
        cell_str = '|'.join(cell_data_list)
        
        # 对首行和首列做截断，以防超出milvus字段字数限制
        # 此外，embedding也无法处理512个token以上的输入
        q_row_col = f'表格首行：{first_row[:256]}\n表格首列：{first_col[:256]}'
        q_cell = f'表格文本：{cell_str[:512]}'
        for q in [q_row_col, q_cell]:
            q_md5 = generate_md5(q)
            # 将(doc_id, doc_text)放入到embedding queue中，执行入mivlus流程
            await emb_queue.put((q_md5, q))
            # 做成表格原子数据，执行入es流程
            table_result[q_md5] = q
        return table_result


    async def table_llm_info(self, emb_queue:asyncio.Queue, before_context:str, table_md:str, after_context:str):
        table_result = dict()
        # TODO: 使用prompt模板管理模组来管理prompt
        table_prompt = """你是一位专注于数据分析和Markdown格式的专家，擅长从表格数据中提取关键信息，并能够根据这些信息生成一系列详细的问题。请根据Markdown格式的表格数据，从中抽取出多个具体的、针对性强的问题。

要求：
1. 生成的问题应涵盖表格的所有列和行，确保全面性；
2. 生成跨列或跨行的综合性问题，体现表格数据之间的关联；
3. 问题的数量控制在2个;
4. 直接按照以下格式输出：
Q1：[具体问题1]
Q2：[具体问题2]

表格上文：{}

表格内容：
{}

表格下文：{}

请严格按照要求对表格进行全面问题抽取，确保问题质量充足且涵盖表格的各个方面。"""
        prompt = table_prompt.format(before_context, table_md, after_context)
        llm_output = await self.llm.send_llm(prompt)
        question_list = llm_output.split("\n")
        for question in question_list:
            question_rag = re.search(r"Q\d+[：|\:](.*)", question)
            if not question_rag:
                continue
            q = question_rag.group(1)
            q_md5 = generate_md5(q)
            # 将(doc_id, doc_text)放入到embedding queue中，执行入mivlus流程
            await emb_queue.put((q_md5, q))
            # 做成表格原子数据，执行入es流程
            table_result[q_md5] = q
        return table_result
            

    async def deal_table(
            self, 
            table_queue:asyncio.Queue, 
            emb_queue:asyncio.Queue
        ):
        table_results = []
        while True:
            await asyncio.sleep(0.1)
            table_info: GraphInfo = await table_queue.get()
            if table_info is None:
                table_queue.task_done()
                break
            # 构造表格文本块，通过拼接上下文的方式，确保表格信息的完整性
            domain, file_name, before, table, after, agg_index, index, atom_index, url = (
                table_info.domain,
                table_info.file_name,
                table_info.before_context,
                table_info.table_md,
                table_info.after_context,
                table_info.agg_index,
                table_info.index,
                table_info.atom_index,
                table_info.url
            )

            tasks = []
            tasks.append(self.table_addon_info(emb_queue, table))
            tasks.append(self.table_llm_info(emb_queue, before, table, after))
            info, llm_info = await asyncio.gather(*tasks)
            info.update(llm_info)
            for doc_id, question in info.items():                                                
                table_results.append(TableChunk(
                    domain=domain,
                    file_name=file_name,
                    doc_id=doc_id,
                    answer=f'{before}\\n\n{table}\n\n{after}'[:5000],    # 进行字数限制，
                    agg_index=agg_index,
                    index=index,
                    atom_index=atom_index,
                    question=question,
                    url=url
                ).model_dump())
            table_queue.task_done()
        # 处理完table，代表需embeding的队列结束
        await emb_queue.put(None)
        logging.info(f'Table parsed, count: {len(table_results)}')
        return table_results


    async def deal_atom(
            self, 
            atom:dict, 
            domain:str, 
            file_name:str, 
            emb_queue:asyncio.Queue, 
            before:str,
            text_results:list, 
            graphs:Dict[str, GraphInfo], 
            temp_graphs:list
        ):
        """atom对象形如：
        {
            'atom_index': atom_index,
            'index': chunk_index,
            'agg_index': belong_index,
            'answer': chunk,
            'chunk_type': 'atom_text'/'table_md'/'image_md'
            'url': url
        }
        """
        # 取出对应字段值
        atom_index = atom['atom_index']
        index = atom['index']
        agg_index = atom['agg_index']
        answer = atom['answer']
        chunk_type = atom['chunk_type']
        url = atom['url']

        if chunk_type == 'atom_text':
            before = answer
            doc_id = generate_md5(before)
            atom_data = AtomChunk(
                domain=domain,
                file_name=file_name,
                doc_id=doc_id,
                answer=answer,
                agg_index=agg_index,
                index=index,
                atom_index=atom_index
            )
            # 送去做embedding和写入milvus
            await emb_queue.put((doc_id, answer))
            text_results.append(atom_data.model_dump())
            # 如果之前缓存了图表块，则当前文本块是图表块的下文
            if temp_graphs:
                for graph_id in temp_graphs:
                    graph = graphs[graph_id]
                    graph.after_context = before
                # 清空缓存的图表
                temp_graphs = []
        elif chunk_type == 'table_md':
            chunk_type = 'table'
            # 在graphs字典中创建GraphInfo实例
            graphs[atom_index] = GraphInfo(
                domain=domain,
                before_context=before,
                table_md=answer,
                # 此时还不知道after_context
                file_name=file_name,
                agg_index=agg_index,
                index=index,
                atom_index=atom_index,
                url=url,
                chunk_type=chunk_type
            )
            # 向temp_graphs中追加atom_index，表示待获取下文
            temp_graphs.append(atom_index)
        elif chunk_type == 'image_md':
            chunk_type = 'image'
            # TODO: 暂不处理图片OCR的结果
            pass
        else:
            raise NotImplementedError(f'Unsupported chunk_type: {chunk_type}')
        return before, text_results, graphs, temp_graphs

    async def build_atom(
            self, 
            domain:str, 
            file_name:str, 
            atom_chunks:list, 
            emb_queue:asyncio.Queue, 
            table_queue:asyncio.Queue
        ):
        # 特殊处理：提取图表上下文用LLM构造问题
        # TODO: 对图表的操作比较重，可考虑放到Chunking类中或独立出来
        before = ''
        temp_graphs = []
        graphs: Dict[str, GraphInfo] = dict()
        text_results = []

        for atom in atom_chunks:
            before, text_results, graphs, temp_graphs = await self.deal_atom(
                atom, domain, file_name, emb_queue, before, text_results, graphs, temp_graphs
            )

        for k, graph in graphs.items():
            if graph.chunk_type == 'table':
                await table_queue.put(graph)
            elif graph.chunk_type == 'image':
                # TODO: 对图片的处理
                pass
            else:
                raise NotImplementedError(f'Unsupported chunk_type: {graph.chunk_type}')
            
        # 添加结束信号
        await table_queue.put(None)
        logging.info(f'atom_text parsed: {file_name}, count: {len(text_results)}')
        return text_results
        

    async def process(self, domain:str, file_name:str, agg_chunks:list, atom_chunks:list):
        emb_queue = asyncio.Queue()
        table_queue = asyncio.Queue()

        preprocessor = InsertPreprocessing(self.pool, self.llm, self.embedding)

        tasks = []
        tasks.append(asyncio.create_task(preprocessor.process_agg(domain, file_name, agg_chunks)))
        tasks.append(asyncio.create_task(preprocessor.build_atom(domain, file_name, atom_chunks, emb_queue, table_queue)))
        tasks.append(asyncio.create_task(preprocessor.deal_table(table_queue, emb_queue)))
        tasks.append(asyncio.create_task(preprocessor.to_embedding(file_name, emb_queue, self.embedding.config.batch_size)))
        # await emb_queue.join()
        # await table_queue.join()
        aggs, text_atoms, table_atoms, emb_results = await asyncio.gather(*tasks)
        logging.info(f'Preprocessed: {file_name}')
        return aggs, text_atoms, table_atoms, emb_results
    

if __name__ == "__main__":
    ...