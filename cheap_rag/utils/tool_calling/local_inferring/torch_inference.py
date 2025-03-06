import asyncio
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union, List
from transformers import AutoTokenizer, AutoModel


# Local modules
from configs.config_cls import LocalEmbeddingConfig
from configs.config import LOCAL_EMBEDDING_CONFIG, MACHINE_ID
from utils.helpers import Singleton, AsyncDict, ftimer, SnowflakeIDGenerator
from utils.tool_calling.local_inferring.embedding_utils import get_detailed_instruct, get_task_def_by_task_name_and_type
from modules.task_manager import CoroTaskManager


class LocalEmbedding(metaclass=Singleton):
    __allow_reinitialization = False

    def __init__(self, config: LocalEmbeddingConfig = None):
        self.config = config or LOCAL_EMBEDDING_CONFIG
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)
        self.model = AutoModel.from_pretrained(self.config.model_dir)
        self.device = torch.device(self.config.emb_type)
        self.model.to(self.device)
        self.id_gen = SnowflakeIDGenerator(MACHINE_ID)
        self.task_manager = CoroTaskManager()
        self.pending_tasks = AsyncDict(self.config.semaphore)


    def count_tokens(self, input_text):
        tokens = self.tokenizer(input_text)
        return len(tokens['input_ids'])


    # Mean pool function
    @staticmethod
    def mean_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.no_grad():
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            result = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return result


    # @ftimer
    def embedding(self, input_texts):
        with torch.no_grad():
            # batch_size = 16 may cost 6-8G GPU memory
            batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {key: value.to(self.device) for key, value in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = self.mean_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.tolist()
    

    @staticmethod
    def build_query(query, task_name='HotpotQA', task_type='Retrieval'):
        task_instr = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
        return get_detailed_instruct(task_instr, query)
    

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


if __name__ == '__main__':
    import time
    from utils.helpers import ftimer

    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
    ]

    emb = LocalEmbedding()

    # res = emb.embedding(documents)
    # print(res.shape)

    res = asyncio.run(emb.a_embedding(documents))
    res = torch.tensor(res, dtype=torch.float16, requires_grad=False)
    norm = torch.norm(res, p=2, dim=-1)
    print(res.shape)
    print(norm)

    # print(ftimer(emb.count_tokens)(documents[0]))

