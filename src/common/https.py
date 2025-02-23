# -*- coding: utf-8 -*-
import re
import json
import logging
import requests
import traceback
from typing import List
from configs.vector_database_config import GLOBAL_CONFIG


def send_request_server(url, content, stream=False, headers=None, verify=False, timeout=10):
    """调用大模型生成问题"""
    try:
        if not headers:
            headers = {"Content-Type": "application/json"}
        params = {
            "model": GLOBAL_CONFIG.llm_config['model'],
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
        response = requests.post(url=url, json=params, headers=headers, timeout=GLOBAL_CONFIG.request_timeout)
        answer = response.text
        return answer
        
    except Exception as err:
        logging.error(traceback.format_exc())
        raise "Faild to request server: {}".format(url)

def send_request_backend_server(url, body, token_id, headers=None, timeout=0):
    try:
        # token_id = "eyJhbGciOiJIUzUxMiJ9.eyJhY2NvdW50IjoiZ2FpeGNhZG1pbiIsInJvbGVMaXN0IjoiR0FJWENfTUFOQUdFUl9ST0xFLEdBSVhDX0RBVEFfTUFOQUdFUixHQUlYQ19DSEFUX0FQSV9GSUxFX01BTkFHRVJfUk9MRSIsInVzZXJJZCI6IkQyN0RFMzM3QUEzMDQ2NjE4MzVDNUQ1Rjg3NkNFRjYzIiwiZW1wbG95ZWVJZCI6IkQxOTcxQkZGQUY0RjRCQjA5RDc4NEU0QzE0QkJBMTk1IiwiZW1wbG95ZWVOYW1lIjoi57O757uf566h55CG5ZGYIiwib3JnSWQiOiJENkNEOEUzODk0M0ZCMjExQzdBREFCRjU1QzRFN0I3NCIsIm9yZ0NvZGUiOiIwIiwib3JnTmFtZSI6IuS4reWbveWNl-aWueeUtee9keaciemZkOi0o-S7u-WFrOWPuCIsInRoaXJkU3lzdGVtTmFtZSI6IkdBSVhDIiwic2FwSHJPcmdJZCI6IjdmMzA0ZGUzNTNjMzU4ZWdjZmcyNjdmMDIzZGI4ZjI1Iiwic3lzdGVtTmFtZSI6IkdBSVhDIiwibmFtZUZ1bGxQYXRoIjoi5Lit5Zu95Y2X5pa555S1572R5pyJ6ZmQ6LSj5Lu75YWs5Y-4IiwiZXhwIjoxNzAzMjQ1NjUyLCJyZWZyZXNoRGF0ZSI6MTcwMzEzNzY1Mjk3OSwianRpIjoiMDUwMTE2YTItZGI5Ni00OThkLTkwYmEtMTNlN2JlMmI5YjZmIiwicmVmcmVzaEludGVydmFsIjoxODAwLCJzdWIiOiLns7vnu5_nrqHnkIblkZgiLCJ0aGlyZENlcnQiOiI0QSJ9.6CUigTpXetzIv2IsW8sKUO7X4KP-RUKHokY1LyjJamIcTG7e8JYpFpJGziyePM8AtzpbOaRpMXXfQ_8rwwamMQ"
        if not headers or "X-Access-Token" not in headers:
            headers = {"Content-type": "application/json", "X-Access-Token": token_id}
        response = requests.post(url=url, json=body, headers=headers, verify=False)
        return response
    except Exception as err:
        logging.error(traceback.format_exc())
        raise "Faild to request server: {}".format(url)


def send_request_embed_server(url, content, csid, headers=None, verify=False, timeout=10):
    try:
        if not headers:
            headers = {"Content-Type": "application/json"}
        # 使用Xinference版本
        if GLOBAL_CONFIG.embed_config['use_xinfer']:
            body = {
                "model": GLOBAL_CONFIG.embed_config['model'],
                "input": content,
            }
            response = requests.post(url=url, json=body, headers=headers, verify=verify, timeout=GLOBAL_CONFIG.request_timeout)
            result = response.json()['data']
            result = [rec['embedding'] for rec in result]
        # 兼容旧版本
        else:
            body = {
                "text": content,
                "csid": csid
            }
            response = requests.post(url=url, json=body, headers=headers, verify=verify)
            result = response.json()
        
        if isinstance(content, str):
            content = [content]
        if not result or len(result) != len(content):
            raise "Faild to call embed server."
        return result

    except Exception as err:
        logging.info(f"------------content--------------{content}")
        logging.info(f"------------response--------------{response.text}")
        # with open("./final_result.json", "w", encoding="utf-8") as f:
        #     json.dump(content, f, indent=4, ensure_ascii=False)
        if isinstance(content, str):
            content = [content]
        logging.info(f"------------output--------------{len(content)}")
        logging.error(traceback.format_exc())
        raise "Faild to request server: {}".format(url)
    

def send_request_rerank_server(url, query:str, documents:List[str], top_n:int):
    try:
        body = {
            'model': GLOBAL_CONFIG.rerank_config['model'],
            'query': query,
            'documents': documents,
            'top_n': top_n
        }
        response = requests.post(url=url, json=body, verify=False, timeout=GLOBAL_CONFIG.request_timeout)
        result = response.json()
        indices, scores = [], []
        for rec in result['results']:
            indices.append(rec['index'])
            scores.append(rec['relevance_score'])
        if not indices:
            raise "Faild to call embed server."
        return indices, scores

    except Exception as err:
        logging.info(f"------------content--------------{content}")
        logging.info(f"------------response--------------{response.text}")
        # with open("./final_result.json", "w", encoding="utf-8") as f:
        #     json.dump(content, f, indent=4, ensure_ascii=False)
        if isinstance(content, str):
            content = [content]
        logging.info(f"------------output--------------{len(content)}")
        logging.error(traceback.format_exc())
        raise "Faild to request server: {}".format(url)
    

def send_request_ocr_server(url, content, csid, headers=None, verify=False, timeout=10):
    try:
        if not headers:
            headers = {"Content-Type": "application/json"}
        body = {
            "text": content,
            "csid": csid
        }
        response = requests.post(url=url, json=body, headers=headers, verify=verify, timeout=GLOBAL_CONFIG.request_timeout)
        result = json.loads(response.text).get("result", [])
        if not result or len(result) != len(content):
            raise "Faild to call ll"
        return result

    except Exception as err:
        logging.error(traceback.format_exc())
        raise "Faild to request server: {}".format(url)