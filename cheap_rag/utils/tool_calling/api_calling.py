import io
import asyncio
import aiohttp
import ssl
import base64
import time
import tarfile
import openai
from os import getenv
from urllib.parse import urljoin
from pathlib import Path
from typing import Union, List


# local module
from utils.logger import logger
from utils.helpers import generate_md5
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
                            elif return_type == 'content':
                                result = await resp.read()
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
        self.semaphore = asyncio.Semaphore(self.config.sema_process)


    async def send_ocr(self, pdf_bs64:str, pdf_prefix:str, ocr_api:str=None):
        url = ocr_api or self.config.base_url
        pdf_name = f'{generate_md5(pdf_prefix)}.pdf'
        data = {"file_name": pdf_name, "file_bs64": pdf_bs64, 'token': getenv('LOCAL_OCR_TOKEN')}
        result = await fetch(
            url, 
            data, 
            self.__class__.__name__, 
            self.config.timeout, 
            semaphore=self.semaphore, 
            return_type='content'
        )
        # Extract the responsed tar.gz file
        tar_gz_file = tarfile.open(fileobj=io.BytesIO(result), mode="r:gz")
        cache_dir = self.config.ocr_cache
        extract_path = cache_dir.joinpath(generate_md5(pdf_prefix))
        extract_path.mkdir(parents=True, exist_ok=True)
        # <pdf_name_md5>
        # |--file_name_md5.json
        # |--images
        #     |--image_0.jpg
        #     |--image_1.jpg
        tar_gz_file.extractall(str(extract_path))


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
    

    async def a_embedding(self, content:Union[str, List[str]], api: str = None):
        # Remote local embedding
        body = {
            'documents': content,
            'token': getenv('LOCAL_EMB_TOKEN')
        }
        result = await fetch(
            api, 
            body, 
            self.__class__.__name__, 
            self.config.timeout,
            semaphore=self.semaphore
        )
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
        if self.config.llm_type == 'openai':
            self.client = openai.AsyncOpenAI(
                api_key=self.config.token.get_secret_value(),
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )

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
            "max_tokens": self.config.max_tokens,
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
    

    async def openai_chat(self, content:List[dict]):
        response = await self.client.chat.completions.create(
            messages=content, 
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        content = response.choices[0].message.content
        return content


if __name__ == '__main__':
    ...

