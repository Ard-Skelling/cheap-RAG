import asyncio
import aiohttp
import ssl
import base64
import time
import openai
from urllib.parse import urljoin
from pathlib import Path
from typing import Union, List


# local module
from utils.logger import logger
from configs.config_cls import (
    OcrConfig,
    EmbeddingConfig,
    LlmConfig
)
from configs.config import (
    OCR_CONFIG,
    EMBEDDING_CONFIG,
    LLM_CONFIG
)
from utils.tool_calling.doc_processing import read_file, write_file


async def fetch(
        url:str, 
        data:Union[dict, aiohttp.FormData], 
        cls_name:str, 
        timeout:float, 
        session_kw:dict=None, 
        request_kw:dict=None,
        semaphore:asyncio.Semaphore=None,
        return_type:str='json'
    ):
        session_kw = session_kw or dict()
        request_kw = request_kw or dict()
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        timeout = aiohttp.ClientTimeout(total=timeout)
        semaphore = semaphore or asyncio.Semaphore(1)
        async with semaphore:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout, **session_kw) as session:
                try:
                    if isinstance(data, dict):
                        request_kw.update({
                            'url': url,
                            'json': data
                        })
                    else:
                        request_kw.update({
                            'url': url,
                            'data': data
                        })
                    async with session.post(**request_kw) as resp:
                        if resp.status == 200:
                            if return_type == 'json':
                                result = await resp.json()
                            elif return_type == 'text':
                                result = await resp.text()
                            else:
                                raise ValueError(f'Unpupported return_type: {return_type}')
                            return result
                        else:
                            raise ValueError(f'{cls_name} 请求失败： {resp.status}')
                except asyncio.TimeoutError as e:
                    logger.info(f'{cls_name} 请求超时')
                    raise e
                except aiohttp.ClientError as e:
                    logger.info(f"{cls_name} 请求错误")
                    raise e


class OcrApi:
    def __init__(self, config:OcrConfig=None) -> None:
        self.config = config or OCR_CONFIG
        self.sema_predict = asyncio.Semaphore(self.config.sema_predict)
        self.sema_download = asyncio.Semaphore(self.config.sema_download)


    async def send_ocr(self, pdf_bytes:bytes, pdf_prefix:str, domain=None):
        url = urljoin(f'http://{self.config.host}:{self.config.port}', self.config.predict_api)
        # 使用minio后的OCR接口，启用时间2024-12-06
        if self.config.use_minio:
            document = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_name = f'{pdf_prefix}.pdf'
            data = {"collection_name":domain, "file": document, "file_name": pdf_name}
            result = await fetch(url, data, self.__class__.__name__, self.config.timeout, semaphore=self.sema_predict)
        # 旧OCR接口
        else:
            data = aiohttp.FormData()
            data.add_field(
                name='files',
                value=pdf_bytes,
                filename=f'{pdf_prefix}.pdf',
                content_type="application/pdf"
            )
            result = await fetch(url, data, self.__class__.__name__, self.config.timeout, semaphore=self.sema_predict)
        return result
    

    async def sned_dir(self, input_dir:str, save_dir:str):
        input_dir = Path(input_dir)
        save_dir = Path(save_dir)
        files = input_dir.rglob('*')
        loop = asyncio.get_running_loop()
        for f in files:
            pdf_prefix = f.stem
            suffix = f.suffix
            if suffix not in ['.pdf', '.PDF']:
                continue
            pdf_bytes = await loop.run_in_executor(None, read_file, str(f))
            res = await self.send_ocr(pdf_bytes, pdf_prefix, domain='temp')
            save_fp = save_dir.joinpath(f'{pdf_prefix}.json')
            await loop.run_in_executor(None, write_file, res, str(save_fp))


class EmbeddingApi:
    def __init__(self, config:EmbeddingConfig=None) -> None:
        self.config = config or EMBEDDING_CONFIG
        self.endpoint = self.config.base_url
        self.semaphore = asyncio.Semaphore(self.config.semaphore)
        if self.config.emb_type == 'openai':
            self.client = openai.AsyncOpenAI(
                api_key=self.config.token.get_secret_value(),
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )

    async def send_embedding(self, content:Union[str, List[str]]):
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self.config.model,
            "input": content,
        }
        result = await fetch(
            urljoin(self.endpoint, self.config.api), 
            body, 
            self.__class__.__name__, 
            self.config.timeout,
            request_kw={'headers': headers},
            semaphore=self.semaphore
        )
        result = [rec['embedding'] for rec in result['data']]
        return result


    async def openai_embedding(self, content:Union[str, List[str]]):
        response = await self.client.embeddings.create(input=content, model=self.config.model)
        embeddings = [ele.embedding for ele in response.data]
        return embeddings


class LlmApi:
    def __init__(self, config:LlmConfig=None) -> None:
        self.config = config or LLM_CONFIG
        self.endpoint = self.config.base_url
        self.semaphore = asyncio.Semaphore(self.config.semaphore)

    async def send_llm(self, content:Union[str, List[str]]):
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": content[:25000]
                }
            ],
            "max_tokens": 2048,
            "presence_penalty": 1.1,
            "temperature": 0.01,
        }
        result = await fetch(
            self.endpoint, 
            body, 
            self.__class__.__name__, 
            self.config.timeout,
            request_kw={'headers': headers},
            semaphore=self.semaphore
        )
        return result['choices'][0]['message']['content']


if __name__ == '__main__':
    import time

    docs = [
        "This agent is developed using Peter Attia‘s publicly available contents and is not affiliated with or endorsed by Peter Attia."
    ]

    emb = EmbeddingApi()
    llm = LlmApi()
    st = time.time()
    res = asyncio.run(emb.openai_embedding(docs))
    et = time.time()
    print(et -st)
    print(res)

    ...

