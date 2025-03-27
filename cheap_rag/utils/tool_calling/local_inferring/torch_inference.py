import asyncio
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union, List, Dict
from transformers import AutoTokenizer, AutoModel
from configs.config_cls import LocalEmbeddingConfig
from configs.config import LOCAL_EMBEDDING_CONFIG, MACHINE_ID
from utils.helpers import Singleton, SnowflakeIDGenerator, ftimer, atimer

class LocalEmbedding(metaclass=Singleton):
    __allow_reinitialization = False

    def __init__(self, config: LocalEmbeddingConfig = None):
        self.config = config or LOCAL_EMBEDDING_CONFIG
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)
        self.model = AutoModel.from_pretrained(self.config.model_dir)
        self.device = torch.device(self.config.emb_type)
        self.model.to(self.device)
        self.id_gen = SnowflakeIDGenerator(MACHINE_ID)

        # Use asynchronous queue to store the pending embedding tasks
        self.request_queue = asyncio.Queue()
        # Use dict to store result futures
        self.result_dict: Dict[int, asyncio.Future] = dict()
        # Max batch size in case of GPU OOM.
        # Computing effeciency is depend on \
        # both batch size and GPUs' ability.
        self.max_batch_size = self.config.batch_size  

        # Start the embedding batch processing loop.
        self.batch_task = asyncio.create_task(self.batch_embedding_loop())

    @staticmethod
    def mean_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.no_grad():
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            result = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return result
    
    def count_tokens(self, input_text):
        tokens = self.tokenizer(input_text)
        return len(tokens['input_ids'])


    def embedding(self, input_texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = self.mean_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.tolist()


    async def a_embedding(self, input_texts: Union[str, List[str]], *args, **kwargs):
        """Asynchronous embedding task, join the request queue, and wait for the return result"""
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        # Generate task_id for a certain embedding task
        task_id = self.id_gen.generate_id()
        future = asyncio.get_event_loop().create_future()
        self.result_dict[task_id] = future
        await self.request_queue.put((task_id, input_texts))
        return await future


    async def batch_embedding_loop(self):
        """Use heartbeat mechanism to collect embedding tasks batch.
        Seperate large tasks into batches."""
        while True:
            await asyncio.sleep(self.config.heartbeat)
            batch_requests = []
            # Collect tasks in heartbeat interval
            while not self.request_queue.empty():
                task_id, input_texts = await self.request_queue.get()
                for text in input_texts:
                    batch_requests.append((task_id, text))
            # Divide large task into batch size
            if batch_requests:
                total_texts = len(batch_requests)
                num_batches = (total_texts + self.max_batch_size - 1) // self.max_batch_size
                # Task_id -- future mapping
                task_results = {}
                start_idx = 0
                for i in range(num_batches):
                    # Deal with one batch embedding task
                    end_idx = min(start_idx + self.max_batch_size, total_texts)
                    batch_data = batch_requests[start_idx:end_idx]
                    batch_texts = [item[1] for item in batch_data]
                    embeddings = await asyncio.to_thread(self.embedding, batch_texts)
                    # Collect result for each task
                    for idx, (task_id, _) in enumerate(batch_data):
                        if task_id not in task_results:
                            task_results[task_id] = []
                        task_results[task_id].append(embeddings[idx])
                    start_idx = end_idx
                # Set the full task result for the certain task
                for task_id, results in task_results.items():
                    if task_id in self.result_dict and not self.result_dict[task_id].done():
                        # Set result for task
                        self.result_dict[task_id].set_result(results)


if __name__ == '__main__':
    import time

    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤..."
    ]

    async def main():
        emb = LocalEmbedding()
        st = time.time()
        tasks = [emb.a_embedding(documents), emb.a_embedding(documents)]
        a, b = await asyncio.gather(*tasks)
        et = time.time()
        print(len(a))
        print(len(b))
        print(et - st)

    asyncio.run(main())
