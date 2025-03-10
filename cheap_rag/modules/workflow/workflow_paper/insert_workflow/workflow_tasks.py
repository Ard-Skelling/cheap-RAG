import re
import asyncio
from typing import Union, Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict


# Local modules
from utils.logger import logger
from utils.helpers import generate_md5, atimer
from utils.tool_calling.local_inferring.torch_inference import LocalEmbedding
from utils.tool_calling.api_calling import LlmApi
from utils.tool_calling.markdownnify import markdownify
from modules.workflow.workflow_paper.config import (
    PaperChunkingConfig,
    PaperInsertPreprocessingConfig,
    CHUNKING_CONFIG,
    IPP_CONFIG
)
from modules.workflow.workflow_paper.data_cls import (
    Task,
    Chunk,
    AggChunk,
    AtomChunk,
    TextChunk,
    EquationChunk,
    ImageChunk,
    TableChunk,
    ESAtomData,
    ESAggData,
    ESRawData,
    MilvusData
)


class Chunking:
    def __init__(self, pool: ProcessPoolExecutor, task:Task, config: PaperChunkingConfig = None) -> None:
        self.config = config or CHUNKING_CONFIG
        self.pool = pool
        # 尽量只从self.task中读值，如果要修改，要加进程锁
        self.task = task
        self.embedding = LocalEmbedding()
        self.init_chunking_params()
        self.init_result_dicts()


    def init_result_dicts(self):
        self.chunks = []
        self.agg_mds = defaultdict(list)
        self.agg_page_idx = dict()
        self.agg_type = dict()
        self.atom_chunks = []


    def init_chunking_params(self):
        self.agg_threshold = self.config.agg_size
        self.agg_overlap = self.config.agg_overlap
        half_overlap = int(self.agg_overlap / 2)
        self.agg_enter = self.agg_threshold - half_overlap
        self.agg_exit = self.agg_threshold + half_overlap
        self.atom_threshold = self.config.atom_size


    @staticmethod
    def has_many_dots(chunk: str):
            """Recognize the table contents."""
            crit_0 = chunk.count('·') >= 5 or chunk.count('·') / len(chunk) >= 0.3
            crit_1 = chunk.count('.') >= 5 or chunk.count('.') / len(chunk) >= 0.3
            return crit_0 or crit_1

    @staticmethod
    def replace_nonsense(text, is_title=False):
        """Handle meaningless characters in text.
        Milvus will recognize some characters as more than one character."""
        if is_title:
            text = re.sub('[^0-9a-zA-Z一-龥 \.·,，。]', '', text)
        text = re.sub(r'···+', '···', text)
        text = re.sub(r'。。+', '。。', text)
        text = re.sub(r'\.\.\.+', '...', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


    @staticmethod
    def standardize_text(text:str):
        # Delete # characters to save tokens
        text = re.sub('^#+', '', text.strip())
        is_contents = Chunking.has_many_dots(text)
        if is_contents:
            text = Chunking.replace_nonsense(text)
        else:
            # Complete sentence
            text = re.sub('(?<=[^.?!"。？！”])\n', ' ', text)
            text = re.sub(r' +', ' ', text)
        return text


    @staticmethod
    def build_text_chunk(file_name:str, agg_index:int, chunk_index:int, block:dict, has_title:bool=None):
        text = block.get('text', '')
        # Filter table contents
        # text = Chunking.standardize_text(text)
        text_level = block.get('text_level', 0)
        page_idx = block.get('page_idx', 0)
        chunk_type = 'text'
        if not has_title and text_level == 1:
            chunk_type = 'title'
            has_title = True
        # use {file_name}{text}{page_idx} to generate md5 mash
        chunk_id = generate_md5(f'{file_name}{text}{page_idx}')
        chunk = TextChunk(
            chunk_id=chunk_id,
            agg_index=agg_index,
            chunk_index=chunk_index,
            page_index=page_idx,
            text=text.strip(),
            text_level=text_level,
            chunk_type=chunk_type
        )
        return chunk, has_title
    

    @staticmethod
    def build_equation_chunk(file_name:str, agg_index:int, chunk_index:int, block:dict):
        text = block.get('text', '')
        page_idx = block.get('page_idx', 0)
        text_format = block.get('text_format', 'latex')
        # use {file_name}{text}{page_idx} to generate md5 mash
        chunk_id = generate_md5(f'{file_name}{text}{page_idx}')
        chunk = EquationChunk(
            chunk_id=chunk_id,
            agg_index=agg_index,
            chunk_index=chunk_index,
            page_index=page_idx,
            text=text.strip(),
            text_format=text_format
        )
        return chunk
    

    @staticmethod
    def build_image_chunk(file_name:str, agg_index:int, chunk_index:int, block:dict):
        url = block.get('img_path', '')
        caption = block.get('img_caption', '')
        footnote = block.get('img_footnote', '')
        page_idx = block.get('page_idx', 0)
        # use {file_name}{url}{page_idx} to generate md5 mash
        chunk_id = generate_md5(f'{file_name}{url}{page_idx}')
        chunk = ImageChunk(
            chunk_id=chunk_id,
            agg_index=agg_index,
            chunk_index=chunk_index,
            page_index=page_idx,
            url=url,
            caption=caption.strip(),
            footnote=footnote.strip()
        )
        # convert list field inot str
        chunk.list2str()
        return chunk

    
    @staticmethod
    def build_table_chunk(file_name:str, agg_index:int, chunk_index:int, block:dict):
        url = block.get('img_path', '')
        caption = block.get('table_caption', '')
        footnote = block.get('table_footnote', '')
        text = block.get('table_body', '')
        # Convert html to markdown
        text = markdownify(text)
        page_idx = block.get('page_idx', 0)
        # Use {file_name}{url}{page_idx} to generate md5 mash
        chunk_id = generate_md5(f'{file_name}{url}{page_idx}')
        chunk = TableChunk(
            chunk_id=chunk_id,
            agg_index=agg_index,
            chunk_index=chunk_index,
            page_index=page_idx,
            url=url,
            caption=caption.strip(),
            footnote=footnote.strip(),
            text=text.strip()
        )
        # convert list field inot str
        chunk.list2str()
        return chunk
    

    def build_chunk(
            self, 
            agg_index:int,
            chunk_index:int, 
            file_name:str, 
            block:dict, 
            has_title:bool=False
        ):
        chunk_type = block.get('type', 'text')
        if chunk_type == 'text':
            chunk, has_title = self.build_text_chunk(file_name, agg_index, chunk_index, block, has_title=has_title)
        elif chunk_type == 'equation':
            chunk = self.build_equation_chunk(file_name, agg_index, chunk_index, block)
        elif chunk_type == 'image':
            chunk = self.build_image_chunk(file_name, agg_index, chunk_index, block)
        elif chunk_type == 'table':
            chunk = self.build_table_chunk(file_name, agg_index, chunk_index, block)
        else:
            raise NotImplementedError(f'Unsupported chunk_type; {chunk_type}')
        # Add raw chunk into self.chunks
        if chunk.chunk_type in {'title', 'text'}:
            if chunk.text:
                self.chunks.append(chunk)
        else:
            self.chunks.append(chunk)
        return chunk, has_title
    

    def build_general_atom(self, file_name:str, agg_index:int, text:str, raw_type=''):
        text = text.strip()
        if text:
            atom_id = generate_md5(f'{file_name}{agg_index}{text}')
            self.atom_chunks.append(AtomChunk(
                atom_id=atom_id,
                text=text,
                agg_index=agg_index,
                file_name=file_name,
                raw_type=raw_type
            ))
    

    def build_text_atom(self, file_name:str, cumu_size:int, text:str, agg_index:int):
        # divide atomic segments
        if cumu_size > 2 * self.atom_threshold:
            texts = re.sub('(?<=\w)\. ', '.\n', text)
            texts = [t.strip() for t in texts.split('\n')]
            cumu = 0
            temp_t = ''
            for t in texts:
                tokens = self.embedding.count_tokens(t)
                cumu += tokens
                temp_t = f'{temp_t}{t}\n'
                if cumu >= self.atom_threshold:
                    self.build_general_atom(file_name, agg_index, temp_t)
                    cumu = 0
                    temp_t = ''
            # append the last segment
            if temp_t:
                self.build_general_atom(file_name, agg_index, temp_t)
        else:
            self.build_general_atom(file_name, agg_index, text)


    def build_image_atom(self, file_name:str, agg_index:int, chunk:ImageChunk, type_prefix='image'):
        for raw_type in ['caption', 'footnote']:
            text = getattr(chunk, raw_type, '')
            self.build_general_atom(file_name, agg_index, text, raw_type=f'{type_prefix}_{raw_type}')


    def build_table_atom(self, file_name:str, agg_index:int, chunk:TableChunk):
        self.build_image_atom(file_name, agg_index, chunk, type_prefix='table')
        self.build_general_atom(file_name, agg_index, chunk.text, raw_type='table_body')

    
    def build_atom(self, file_name:str, agg_index, chunk:Chunk, cumu_size=0, cache_text=''):
        chunk_type = chunk.chunk_type
        if chunk_type == 'text':
            self.build_text_atom(file_name, cumu_size, cache_text, agg_index)
        elif chunk_type == 'image':
            self.build_image_atom(file_name, agg_index, chunk)
        elif chunk_type == 'table':
            self.build_table_atom(file_name, agg_index, chunk)
        else:
            self.build_general_atom(file_name, agg_index, getattr(chunk, 'text', ''))


    def build_agg(self, file_name):
        for agg_index, agg_cache in self.agg_mds.items():
            page_index = self.agg_page_idx[agg_index]
            agg_t = self.agg_type[agg_index]
            agg_text = '\n\n'.join([agg.strip() for agg in agg_cache])
            yield AggChunk(
                agg_id=generate_md5(f'{file_name}{agg_index}'),
                agg_index=agg_index,
                page_index=page_index,
                text=agg_text,
                chunk_type=agg_t
            )


    def ignore_exam(self, chunk:Union[Chunk, TextChunk], ignore_count=0):
        # Ignore the REFERENCES block and 1 successor
        if chunk.chunk_type == 'text' and 'references' in chunk.text.strip().lower():
            ignore_count = 2
        # Ignore the empty text block
        elif chunk.chunk_type == 'text' and not chunk.text.strip():
            ignore_count = 1
        else:
            pass
        if ignore_count > 0:
            ignore = True
        else:
            ignore = False
        return ignore, ignore_count


    def chunk2markdown(self, chunk:Union[TextChunk, ImageChunk, TableChunk]):
        chunk_type = chunk.chunk_type
        if chunk_type == 'image':
            url = chunk.url
            caption = chunk.caption
            footnote = chunk.footnote
            temp_res = ''
            if caption:
                temp_res = f'{temp_res}{caption}\n\n'
            # TODO: Image OCR text
            if url:
                temp_res = f'{temp_res}Image url: {url}\n\n'
            if footnote:
                temp_res = f'{temp_res}{footnote}'
            return temp_res.strip()
        elif chunk_type == 'table':
            url = chunk.url
            caption = chunk.caption
            footnote = chunk.footnote
            text = chunk.text
            temp_res = ''
            if caption:
                temp_res = f'{temp_res}{caption}\n\n'
            if text:
                if self.config.show_table_md:
                    temp_res = f'{temp_res}{text}\n\n'
                else:
                    temp_res = f'{temp_res}<table>{chunk.url}</table>\n\n'
            if url:
                temp_res = f'{temp_res}Table url: {url}\n\n'
            if footnote:
                temp_res = f'{temp_res}{footnote}'
            return temp_res.strip()
        else:
            return chunk.text.strip()
        

    def cache_agg_seg(self, agg_index, chunk_type, chunk_md, page_index):
        # Cache the temperary aggregated chunk segment
        self.agg_mds[agg_index].append(chunk_md)
        # Use the first page index as aggregated chunk's page index
        if agg_index not in self.agg_page_idx:
            self.agg_page_idx[agg_index] = page_index
            # Set aggregated chunk type as special type
            if chunk_type in {'title', 'summary', 'outline'}:
                self.agg_type[agg_index] = chunk_type
            # Set aggregated chunk type as normal text type
            else:
                self.agg_type[agg_index] = 'text'


    def cumulate_agg(
        self, 
        belong_index: int,
        agg_index: int,
        page_index: int,
        agg_cumu: int,
        chunk: Chunk
    ):
        # Aggregate chunk fields into text
        chunk_type = chunk.chunk_type
        chunk_md = self.chunk2markdown(chunk)
        chunk_size = self.embedding.count_tokens(chunk_md)
        # 将当前片追加到当前agg_index
        self.cache_agg_seg(belong_index, chunk_type, chunk_md, page_index)
        agg_cumu += chunk_size
        if agg_cumu > self.agg_enter:
            # 当前片已进入overlap区间，需加入下一聚合片
            next_index = agg_index + 1
            self.cache_agg_seg(next_index, chunk_type, chunk_md, page_index)
            if agg_cumu > self.agg_threshold:
                # 当前片超越overlap的中间点，需更新用于原子片的belong_index变量
                belong_index = next_index
                if agg_cumu > self.agg_exit:
                    # 当前片已越过overlap，需重启用新的agg_index
                    agg_index = next_index
                    agg_cumu -= self.config.agg_size
        return chunk_md, chunk_size, belong_index, agg_index, agg_cumu


    def cumulate_atom(
        self,
        file_name: str,
        atom_cumu: int,
        temp_atom: str,
        chunk: Union[Chunk, TextChunk],
        chunk_md: str,
        belong_index: int,
        chunk_size,
        is_last
    ):
        # Exam to build atomic chunk or not
        # Non-text chunk
        if chunk.chunk_type not in {'title', 'text'}:
            self.build_atom(file_name, belong_index, chunk)
        # Text chunk
        else:
            # Build atomic chunk before header-content block
            if chunk.text_level != 0 and atom_cumu > 0.7 * self.atom_threshold:
                self.build_text_atom(file_name, chunk_size, temp_atom, belong_index)
                atom_cumu = 0
                temp_atom = ''
            # Exam atomic chunk threshold
            atom_cumu += chunk_size
            temp_atom = f'{temp_atom}{chunk_md}\n\n'
            if atom_cumu >= self.atom_threshold:
                self.build_text_atom(file_name, chunk_size, temp_atom, belong_index)
                atom_cumu = 0
                temp_atom = ''
        return atom_cumu, temp_atom


    def chunking(self, file_name, document: List[dict]):
        has_title = False
        # Initiate the index of agg_chunks
        agg_index = 1   
        # Cumulated tokens of cached chunk_md to build agg_chunk 
        agg_cumu = 0    
        # The index of agg_chunk that the atomic chunk belongs to
        belong_index = 1 
        # Cumulated tokens of atomic chunk
        atom_cumu = 0    
        temp_atom = ''

        chunks_len = len(document)
        # Used to identify the title of the document
        has_title = False
        ignore_count = 0

        for i, block in enumerate(document):
            is_last = i == chunks_len - 1
            chunk, has_title = self.build_chunk(
                belong_index, i + 1, file_name, block, has_title
            )

            # Ignore chunk exam
            ignore, ignore_count = self.ignore_exam(chunk, ignore_count)
            if ignore:
                if chunk.chunk_type == 'text':
                    ignore_count = ignore_count - 1
                    # Ignore chunk and it's successors if applicable
                    continue
                else:
                    # Reset ignore counter for non-text chunk
                    ignore_count = 0

            # Build aggregated chunk
            chunk_md, chunk_size, belong_index, agg_index, agg_cumu = self.cumulate_agg(
                belong_index=belong_index,
                agg_index=agg_index, 
                page_index=chunk.page_index,
                agg_cumu=agg_cumu, 
                chunk=chunk
            )

            # 拼接和做成原子片
            atom_cumu, temp_atom = self.cumulate_atom(
                file_name=file_name,
                atom_cumu=atom_cumu, 
                temp_atom=temp_atom,
                chunk=chunk,
                chunk_md=chunk_md,
                belong_index=belong_index,
                chunk_size=chunk_size,
                is_last=is_last
            )

        result = {
            'chunks': self.chunks,
            'agg_chunks': [agg for agg in self.build_agg(file_name)],
            'atom_chunks': self.atom_chunks
        }
        return result


class InsertPreprocessing:
    def __init__(self, pool:ProcessPoolExecutor, llm:LlmApi, config:PaperInsertPreprocessingConfig = None) -> None:
        self.config = config or IPP_CONFIG
        self.pool = pool
        self.llm = llm
        self.embedding = LocalEmbedding()
    

    def process_chunks(self, file_name: str, chunks: Union[List[Chunk], List[AggChunk]], data_cls:Union[ESRawData, ESAggData]):
        # TODO: AI info enhancement
        chunks = [data_cls(file_name=file_name, **chunk.model_dump()).model_dump(by_alias=True) for chunk in chunks]
        return chunks
    

    async def process_atoms(self, table_queue:asyncio.Queue, emb_queue:asyncio.Queue, atom_chunks: List[AtomChunk]):
        for atom in atom_chunks:
            # Table information enhancement producer
            if atom.raw_type == 'table_body':
                # Consumed in self.process_table_atom
                await table_queue.put(atom)
            await emb_queue.put(atom)
        # Terminate queued task.
        await table_queue.put(None)
        await emb_queue.put(None)
        return atom_chunks


    async def to_embedding(self, file_name, emb_queue:asyncio.Queue, batch_size:int):
        emb_results = dict()
        doc_ids = []
        docs = []
        count = 0
        while True:
            await asyncio.sleep(0.1)
            atom: AtomChunk = await emb_queue.get()
            if atom is None:
                emb_queue.task_done()
                break
            doc_ids.append((atom.atom_id, atom.agg_index))
            docs.append(atom.text)
            count += 1
            if count == batch_size:
                res = await self.embedding.a_embedding(docs)
                res = dict(zip(doc_ids, res))
                emb_results.update(res)
                doc_ids = []
                docs = []
                count = 0
            emb_queue.task_done()
        # Deal with residual data
        if doc_ids:
            res = await self.embedding.a_embedding(docs)
            res = dict(zip(doc_ids, res))
            emb_results.update(res)
        # TODO: Use async generator
        emb_results = [MilvusData(atom_id=doc_id[0], agg_index=doc_id[1], vec=vec, file_name=file_name) for doc_id, vec in emb_results.items()]
        logger.info(f'Embedding finished: {file_name}, count: {len(emb_results)}')
        return emb_results
    

    async def process_table_atom(self, file_name, table_queue: asyncio.Queue, emb_queue: asyncio.Queue):
        results = []
        while True:
            await asyncio.sleep(0.1)
            table_chunk: AtomChunk = await table_queue.get()
            if table_chunk is None:
                table_queue.task_done()
                break
            tasks = []
            tasks.append(self.table_llm_info(emb_queue, table_chunk))
            tasks.append(self.table_addon_info(file_name, emb_queue, table_chunk))
            info, llm_info = await asyncio.gather(*tasks)
            info.update(llm_info)
            for atom in info.values():
                await emb_queue.put(atom)
                results.append(atom)
            table_queue.task_done()
        return results
    

    async def table_llm_info(
        self, 
        emb_queue: asyncio.Queue, 
        table_chunk: AtomChunk
    ):
        # TODO: AI info enhancment
        return dict()
    

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


    async def table_addon_info(
        self, 
        file_name,
        emb_queue: asyncio.Queue, 
        table_chunk: AtomChunk
    ):
        table_result = dict()
        table_md = table_chunk.text
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
        q_row_col = f'Table first row: {first_row[:256]}\nTable first column: {first_col[:256]}'
        q_cell = f'Table cell texts: {cell_str[:512]}'
        for q in [q_row_col, q_cell]:
            q_md5 = generate_md5(f'{file_name}{table_chunk.atom_id}{q}')
            # 将(doc_id, doc_text)放入到embedding queue中，执行入mivlus流程
            atom = AtomChunk(
                atom_id=q_md5,
                text=q,
                agg_index=table_chunk.agg_index,
                file_name=file_name,
                raw_type=table_chunk.raw_type
            )
            await emb_queue.put(atom)
            # 做成表格原子数据，执行入es流程
            table_result[q_md5] = atom
        return table_result


    @atimer
    async def process(
        self, 
        file_name: str, 
        chunks: List[AggChunk], 
        agg_chunks: List[AggChunk], 
        atom_chunks: List[AtomChunk]
    ):
        emb_queue = asyncio.Queue()
        table_queue = asyncio.Queue()

        tasks = []
        process_chunks = asyncio.to_thread(self.process_chunks, file_name, chunks, ESRawData)
        tasks.append(asyncio.create_task(process_chunks))
        process_aggs = asyncio.to_thread(self.process_chunks, file_name, agg_chunks, ESAggData)
        tasks.append(asyncio.create_task(process_aggs))
        tasks.append(asyncio.create_task(self.process_atoms(table_queue, emb_queue, atom_chunks)))
        tasks.append(asyncio.create_task(self.process_table_atom(file_name, table_queue, emb_queue)))
        tasks.append(asyncio.create_task(self.to_embedding(file_name, emb_queue, self.embedding.config.batch_size)))
        chunks, aggs, atoms, table_atoms, emb_results = await asyncio.gather(*tasks)
        chunks: List[dict] = chunks
        aggs: List[dict] = aggs
        atoms: List[AtomChunk] = atoms
        table_atoms: List[AtomChunk] = table_atoms
        emb_results: List[MilvusData] = emb_results
        es_atoms = atoms + table_atoms
        es_atoms = [ESAtomData(file_name=file_name, **atom.model_dump()) for atom in es_atoms]
        atoms = [atom.model_dump(by_alias=True) for atom in es_atoms]
        emb_results = [emb.model_dump(by_alias=True) for emb in emb_results]
        logger.info(f'Preprocessed: {file_name}')
        return chunks, aggs, atoms, emb_results
    
