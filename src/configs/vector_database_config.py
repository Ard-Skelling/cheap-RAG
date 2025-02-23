# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
from pathlib import Path
from configs import PublicEngineConfig
# from dotenv import load_dotenv

# load_dotenv('/home/RAG/oct28/.env')


BASEPATH = Path(__file__).parent.parent

VD_HOST = os.getenv('VD_HOST') or '192.168.210.113'
VD_PORT = os.getenv('VD_PORT') or 19530
EMB_HOST = os.getenv('EMB_HOST') or '192.168.210.128'
EMB_PORT = os.getenv('EMB_PORT') or 19998
LLM_HOST = os.getenv('LLM_HOST') or '192.168.210.117'
LLM_PORT = os.getenv('LLM_PORT') or 1060
OCR_HOST = os.getenv('OCR_HOST') or '192.168.210.128'
OCR_PORT = os.getenv('OCR_PORT') or 42008
MINIO_HOST = os.getenv('MINIO_HOST') or '192.168.210.113'
MINIO_PORT = os.getenv('MINIO_PORT') or 9000
MINIO_ACCESS = os.getenv('MINIO_ACCESS') or 'minioadmin'
MINIO_SECRET = os.getenv('MINIO_SECRET') or 'minioadmin'
ES_HOST = os.getenv('ES_HOST') or '192.168.210.113'
ES_PORT = os.getenv('ES_PORT') or 9200
ES_USER = os.getenv('ES_USER') or 'elastic'
ES_PWD = os.getenv('ES_PWD') or ''
USE_XINFER_EMBED = False if os.getenv('USE_XINFER_EMBED') in {'false', 'False', 0, '0', False} else True
USE_XINFER_RERANK = False if os.getenv('USE_XINFER_RERANK') in {'false', 'False', 0, '0', False} else True


class VecDataModuleParameter(PublicEngineConfig):
    """向量模块相关参数配置"""

    # pymilvus 配置
    vecdata_config={
        "ip": VD_HOST,  # milvus 部署服务器的ip
        "port": VD_PORT,
        "index_type": "GPU_IVF_FLAT",  # 搜索索引类型，cpu上会自动降级为IVF_FLAT
        "metric_type": "IP",  # 向量计算方式 IP, COSIN, L2
    }

    # 文本切分相关配置
    spliter_config={
        "max_length": 500,
        "min_count": 20,
        "threshold": 5000,
        "chunk_size": 1000,
        "overlap": 300,
        "agg_length": 20000
    }

    # 向量模型相关配置
    embed_config={
        # "url": f"http://{EMB_HOST}:{EMB_PORT}/v1/model/embedEngine",
        "url": f"http://{EMB_HOST}:{EMB_PORT}/v1/embeddings",
        "thread_num": 3,
        "use_xinfer": USE_XINFER_EMBED,
        "model": "rgzn_bge_model",
        "batch_size": 16
    }

    # 大模型配置
    llm_config={
        "prompt": """你现在是一个问答对构造师, 现在我给你提供一段答案, 需要你来根据答案生成问题, 但是要求如下：
1. 生成的问题要简洁精练,要能体现出对答案的重点描述。
2. 生成的问题可以是一个或者多个,但是答案之间要以1.xxx2.xxx格式输出。
现在给你的答案是：{}\n。""",
        "url": f"http://{LLM_HOST}:{LLM_PORT}/v1/chat/completions",
        "model": "llm_72b"
    }

    # ocr 配置
    ocr_config = {
        'url': f"http://{OCR_HOST}:{OCR_PORT}/predict",
        'timeout': 99999,
        'use_minio': True
    }

    # 是否启用开发模式。开发模式会保留一些中间变量到.ocrs/目录，并打印更详细的中间变量
    debug_mode = True if os.getenv('IS_DEBUG') in ['true', 'True', 1, '1', True] else False

    debug_config = {
        'host': '127.0.0.1',
        'port': 20002
    }

    minio_config = {
        'host': MINIO_HOST,
        'port': 29000 if debug_mode else MINIO_PORT,
        'ak': 'jc_test_minio' if debug_mode else MINIO_ACCESS,
        'sk': 'jc_test_minio' if debug_mode else MINIO_SECRET,
        'max_workers': 1024,
        'bucket': 'file-meta',
        'bucket_ocr': 'file-ocr'
    }

    es_config = {
        'host': ES_HOST,
        'port': ES_PORT,
        'user': ES_USER,
        'pwd': ES_PWD
    }

    # 重排模型相关配置
    rerank_config={
        "url": f"http://{EMB_HOST}:{EMB_PORT}/v1/rerank",
        "thread_num": 3,
        "use_xinfer": USE_XINFER_RERANK,
        "model": "bge-reranker-large"
    }

    request_timeout = 99999


GLOBAL_CONFIG = VecDataModuleParameter()