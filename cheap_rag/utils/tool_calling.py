import sys
import asyncio
import aiohttp
import ssl
import json
import subprocess
import shutil
import base64
import logging
import time
from urllib.parse import urljoin
from pathlib import Path
from typing import Union, List


BASE_PATH = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_PATH))


# local module
from configs.config_v2.config_cls import (
    FileConvertConfig,
    OcrConfig,
    EmbeddingConfig,
    LlmConfig
)
from configs.config_v2.config import (
    FILE_CONVERT_CONFIG,
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
                            else:
                                raise ValueError(f'Unpupported return_type: {return_type}')
                            return result
                        else:
                            raise ValueError(f'{cls_name} 请求失败： {resp.status}')
                except asyncio.TimeoutError as e:
                    logging.info(f'{cls_name} 请求超时')
                    raise e
                except aiohttp.ClientError as e:
                    logging.info(f"{cls_name} 请求错误")
                    raise e


def read_file(file_path:str):
    fp = Path(file_path)
    suffix = fp.suffix.lower()
    if suffix in {'.pdf', '.doc', '.docx'}:
        with open(file_path, 'rb') as f:
            return f.read()
    elif suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise NotImplementedError(f'Unsupported file format!\nReading_file: {file_path}')


def write_file(data, file_path:str):
    fp = Path(file_path)
    suffix = fp.suffix.lower()
    if suffix in {'.pdf', '.doc', '.docx'}:
        with open(file_path, 'wb') as f:
            f.write(data)
    elif suffix == '.json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    else:
        raise NotImplementedError(f'Unsupported file format!\nWriting_file: {file_path}')


class FileConverter:
    def __init__(self, config:FileConvertConfig=None) -> None:
        self.task_type = ''
        self.config = config or FILE_CONVERT_CONFIG
        self.semaphore = asyncio.Semaphore(self.config.num_workers)
        self.preheat_task = None

    
    async def preheat_libreoffice(self):
        """启动LibreOffice后台进程并保持活跃状态"""
        try:
            # 启动 LibreOffice --headless 以预热
            logging.info("Starting LibreOffice to preheat...")
            cmd = ['soffice', '--headless', '--writer', '--convert-to', 'pdf', '--outdir', '/.cache', '--nologo']
            
            # 启动子进程，并等待其执行
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 等待子进程完成初始化
            stdout, stderr = await process.communicate()
            err_mesage = stderr.decode('utf-8')
            if 'error' in err_mesage.lower():
                raise Exception(f"Failed to preheat LibreOffice: {stderr.decode('utf-8')}")
            
            logging.info("LibreOffice preheated successfully.")
        
        except Exception as e:
            logging.error(f"Error in preheating LibreOffice: {str(e)}")


    async def initialize(self):
        """初始化FileConverter并预热LibreOffice"""
        while True:
            try:
                self.preheat_task = asyncio.create_task(self.preheat_libreoffice())
                break
            except Exception as err:
                await asyncio.sleep(0.1)
                print('aaaaa')
                logging.error(repr(err))


    async def _conver_file(self, doc_file_path, output_dir, file_format):
        cmd = ['soffice', '--headless', '--convert-to', file_format, '--outdir', output_dir, doc_file_path]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,  # 捕获标准输出
            stderr=asyncio.subprocess.PIPE   # 捕获标准错误
        )
        # 等待子进程完成并获取输出
        stdout, stderr = await process.communicate()
        return stdout, stderr
    
    
    def convert_file(self, doc_file_path, output_dir, file_format):
        cmd = ['soffice', '--headless', '--convert-to', file_format, '--outdir', output_dir, doc_file_path]
        subprocess.run(cmd, shell=False, check=False, capture_output=True, timeout=self.config.timout)
        logging.info(f'Sucessed to convert docx to pdf: {Path(doc_file_path).name}')


    async def _a_convert_file(self, doc_file_path, output_dir, file_format):
        # await self.preheat_task
        # 创建子进程，在子进程运行系统命令，由系统层面自动实现并行调度
        async with self.semaphore:
            cmd = ['soffice', '--headless', '--convert-to', file_format, '--outdir', output_dir, doc_file_path]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,  # 捕获标准输出
                stderr=asyncio.subprocess.PIPE   # 捕获标准错误
            )
            # 等待子进程完成并获取输出
            stdout, stderr = await process.communicate()
            # 返回标准输出和标准错误
        stdout_message = stdout.decode('utf-8')
        logging.info(stdout_message)
        error_message = stderr.decode('utf-8')
        if 'error' in error_message.lower():
            file_name = Path(doc_file_path).name
            raise RuntimeError(f'Convert file failed: {file_name}\n{error_message}')


    async def a_convert_file(self, doc_file_path, output_dir, file_format, retry=5):
        file_name = Path(doc_file_path).name
        tgt_fp = Path(output_dir) / Path(file_name).with_suffix('.pdf')
        err = RuntimeError(f'LibreOffice run successful, but no output file when converting {file_name}')
        for i in range(retry):
            try:
                res = await asyncio.wait_for(
                    self._a_convert_file(doc_file_path, output_dir, file_format), 
                    timeout=self.config.timout
                )
                if tgt_fp.exists():
                    return res
                else:
                    await asyncio.sleep(0.5)
            except Exception as err:
                logging.warning(f'Error occurred when converting file: {file_name}. {i} Retrying...\n{repr(err)}')
        raise err


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
        self.endpoint = urljoin(f'http://{self.config.host}:{self.config.port}', self.config.api)
        self.semaphore = asyncio.Semaphore(self.config.semaphore)

    async def send_embedding(self, content:Union[str, List[str]]):
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self.config.model,
            "input": content,
        }
        result = await fetch(
            self.endpoint, 
            body, 
            self.__class__.__name__, 
            self.config.timeout,
            request_kw={'headers': headers},
            semaphore=self.semaphore
        )
        result = [rec['embedding'] for rec in result['data']]
        return result

    

class LlmApi:
    def __init__(self, config:LlmConfig=None) -> None:
        self.config = config or LLM_CONFIG
        self.endpoint = urljoin(f'http://{self.config.host}:{self.config.port}', self.config.api)
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
    # import time


    # docs = [
    #     '中国南方电网公司的综合管理体系的内涵是？',
    #     '综合管理体系的内涵：以系统分析方法和现代企业治理思维对公司管理体系和管理能力进行全面升级与科学管控的理论实践。'
    # ]


    # async def test_api():
    #     doc = '中国南方电网公司的综合管理体系的内涵是？'
    #     count = 0
    #     for i in range(200):
    #         res = await llm.send_llm(doc)
    #         count += len(res)
    #     return count

    
    # # emb = EmbeddingApi()
    # llm = LlmApi()
    # st = time.time()
    # res = asyncio.run(test_api())
    # et = time.time()
    # print(et -st)
    # print(res)

    # fp = '/gpfs/jincheng/csg/RAG/local_dev/022.TB 10101-2018_铁路工程测量规范.pdf'
    # ocr = OcrApi()
    # with open(fp, 'rb') as f:
    #     pdf_bytes = f.read()
    # res = asyncio.run(ocr.send_ocr(pdf_bytes, Path(fp).stem, 'test'))
    # with open('/gpfs/jincheng/csg/RAG/local_dev/ocr_samples/sample_240113.json', 'w') as f:
    #     json.dump(res, f, indent=4, ensure_ascii=False)

    convertor = FileConverter()
    fp = '/gpfs/jincheng/csg/RAG/local_dev/2、技术规范书-车网互动项目.docx'
    odir = '/gpfs/jincheng/csg/RAG/local_dev/ocr_samples'
    fmt = 'pdf'
    res = asyncio.run(convertor.a_convert_file(fp, odir, fmt))
    ...

