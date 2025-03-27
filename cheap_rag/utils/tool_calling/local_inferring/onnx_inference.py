import numpy as np
import asyncio
import onnxruntime as ort
from typing import Union, List
from transformers import AutoTokenizer


# Local modules
from configs.config_cls import LocalEmbeddingConfig
from configs.config import LOCAL_EMBEDDING_CONFIG, MACHINE_ID
from utils.helpers import Singleton, SnowflakeIDGenerator, AsyncDict, ftimer
from utils.tool_calling.local_inferring.embedding_utils import (
    get_detailed_instruct,
    get_task_def_by_task_name_and_type
)

from modules.task_manager import CoroTaskManager


class LocalEmbedding(metaclass=Singleton):
    __allow_reinitialization = False
    
    def __init__(self, config: LocalEmbeddingConfig = None):
        self.config = config or LOCAL_EMBEDDING_CONFIG
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)
        if self.config.emb_type == 'cpu':
            self.onnx_session = ort.InferenceSession(
                self.config.model_path, providers=['CPUExecutionProvider']
            )
        elif self.config.emb_type == 'cuda':
            self.onnx_session = ort.InferenceSession(
                self.config.model_path, providers=[
                    'TensorrtExecutionProvider', 
                    'CUDAExecutionProvider'
                ]
            )
        else:
            raise NotImplementedError(f'Unsupported emb_type: {self.config.emb_type}')
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name
        self.id_gen = SnowflakeIDGenerator(MACHINE_ID)
        self.task_manager = CoroTaskManager()
        self.pending_tasks = AsyncDict(self.config.semaphore)

    
    def count_tokens(self, input_text):
        tokens = self.tokenizer(input_text)
        return len(tokens['input_ids'])


    # Mean pool function
    def np_mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray):
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    

    def np_normalize(self, arry, p=2, dim=-1, eps=1e-12):
        # L2 norm
        norm = np.linalg.norm(arry, ord=p, axis=dim, keepdims=True)
        # Avoid divided by 0 error
        norm = np.maximum(norm, eps)
        return arry / norm


    @ftimer
    def embedding(self, input_texts):
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='np')
        encoded_input = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']
        # Make sure to pass both 'input_ids' and 'attention_mask'
        result = self.onnx_session.run([self.output_name], {self.input_name: encoded_input, 'attention_mask': attention_mask})[0]
        result = self.np_mean_pooling(result, attention_mask)
        result = self.np_normalize(result)
        return result
    

    async def a_embedding(self, input_texts: Union[str, List[str]]):
        task_id = self.id_gen.generate_id()
        # Task queue with max length
        await self.pending_tasks.put(task_id, input_texts)
        input_texts = await self.pending_tasks.pop(task_id)
        if not input_texts:
            raise ValueError(f'Task {task_id} is not in ongoing tasks.')
        # Task execution
        task = asyncio.create_task(asyncio.to_thread(self.embedding, input_texts))
        await self.task_manager.add_task(task_id, task)
        result = await self.task_manager.wait_result(task_id, timeout=self.config.timeout)
        return result
    

    @staticmethod
    def build_query(query, task_name='HotpotQA', task_type='Retrieval'):
        task_instr = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
        return get_detailed_instruct(task_instr, query)



if __name__ == '__main__':
    import time

    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
    ]

    emb = LocalEmbedding()
    res = emb.embedding(documents)
    print(res.shape)
    ...