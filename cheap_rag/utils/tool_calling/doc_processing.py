import asyncio
import json
import subprocess
import aiofiles
from pathlib import Path


# local module
from utils.logger import logger
from configs.config_cls import FileConvertConfig
from configs.config import FILE_CONVERT_CONFIG


async def read_file(file_path:str):
    fp = Path(file_path)
    suffix = fp.suffix.lower()
    if suffix in {'.pdf', '.doc', '.docx'}:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
    elif suffix == '.json':
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            content = json.loads(content)
    else:
        raise NotImplementedError(f'Unsupported file format!\nReading_file: {file_path}')
    return content


async def write_file(data, file_path:str):
    fp = Path(file_path)
    suffix = fp.suffix.lower()
    if suffix in {'.pdf', '.doc', '.docx'}:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)
    elif suffix == '.json':
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            content = json.dumps(data, indent=4, ensure_ascii=False)
            await f.write(content)
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
            logger.info("Starting LibreOffice to preheat...")
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
            
            logger.info("LibreOffice preheated successfully.")
        
        except Exception as e:
            logger.error(f"Error in preheating LibreOffice: {str(e)}")


    async def initialize(self):
        """初始化FileConverter并预热LibreOffice"""
        while True:
            try:
                self.preheat_task = asyncio.create_task(self.preheat_libreoffice())
                break
            except Exception as err:
                await asyncio.sleep(0.1)
                print('aaaaa')
                logger.error(repr(err))


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
        logger.info(f'Sucessed to convert docx to pdf: {Path(doc_file_path).name}')


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
        logger.info(stdout_message)
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
                logger.warning(f'Error occurred when converting file: {file_name}. {i} Retrying...\n{repr(err)}')
        raise err
