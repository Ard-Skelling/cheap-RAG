import uuid
import requests
import base64
from typing import Literal, Union, List
from pathlib import Path
from urllib.parse import urljoin


HOST = '127.0.0.1'
PORT = 20002


mac = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
MAC = "-".join([mac[e:e+2] for e in range(0, 11, 2)])
RETRIEVE_API = f'http://{HOST}:{PORT}/v1/model/'
REQUEST_HEADERS = {
    'Content-Type': 'Application/json'
}


def file2base64(file_path):
    with open(file_path, 'rb') as f:
        bytes_data = f.read()
    document = base64.b64encode(bytes_data).decode("utf-8")
    return document


def test_insert(file_path:str, domain:str, file_meta:dict=None, is_enhanced=False):
    '''测试pdf, docx, doc, txt, csv, xslx, xsl, json等文件格式写入向量数据库。
    
    Params:
        file_path(str): 测试文件的本地存储路径。

    Return:
        dict: 写入任务的执行结果。
    '''
    api = urljoin(RETRIEVE_API, 'DataInsertEngine')
    fp = Path(file_path)
    body = {
        'document': file2base64(file_path),
        'domain': domain,
        'file_name': fp.name,
        'csid': MAC,
        'file_meta': file_meta,
        'is_enhanced': is_enhanced
    }
    res = requests.post(api, json=body, headers=REQUEST_HEADERS)
    print(res.json())
    return res.json()


def test_field_search(domain:str, output_fields:List[str]):
    '''测试检索出指定字段去重后的所有值
    
    Params:
        domain(str): 数据集名称
        output_fields(List[str]): 想要查看的字段

    Return:
        dict: 任务的执行结果。
    '''
    url = urljoin(RETRIEVE_API, 'FieldSearchEngine')
    body = {
        'domain': domain,
        'output_fields': output_fields,
        'csid': MAC
    }
    try:
        resp = requests.post(url, json=body, headers=REQUEST_HEADERS)
        print(resp.json())
    except Exception as e:
        raise e
    

def test_file_name_search(domain:str):
    '''测试检索出所有文件名
    
    Params:
        domain(str): 数据集名称

    Return:
        dict: 任务的执行结果。
    '''
    url = urljoin(RETRIEVE_API, 'FileNameSearchEngine')
    body = {
        'domain': domain,
        'csid': MAC,
    }
    try:
        resp = requests.post(url, json=body, headers=REQUEST_HEADERS)
        print(resp.json())
    except Exception as e:
        raise e


def test_search(query:str, 
    domain:str, 
    search_field:Literal['vec', 'slice_vec,' 'q_slice_vec']=None, 
    threshold=0.3, 
    topn=5):
    """请求召回服务，召回与query匹配的一组文本。

    Args:
        query (str): 查询文本
        db_name (str, optional): milvus数据集名称. Defaults to DB_NAME.
        search_field (str, optional): 进行相关度计算的目标字段. Defaults to None.
        threshold (float, optional): 相关度阈值，0-1之间. Defaults to 0.3.
        topn (int, optional): 返回最相关的topk个结果. Defaults to 10. 

    Returns:
        List[dict]: 一组与qury最相关的数据, 形如：
            [
                {'score': 0.9161,
                'question': '安全框架',
                'answer': '',
                'index': 92,
                'file_name': '调研报告0513.pdf',
                'url': 'images/page67_image0.jpg'},
                {'score': 0.6446,
                'question': '涵盖哪些安全风险？',
                'answer': XXXXXX',
                'index': 92,
                'file_name': 'jolwl',
                'url': ''},
            ]
    """
    url = urljoin(RETRIEVE_API, 'DataSearchEngine')
    body = {
        'domain': domain,
        'question': query,
        'search_field': search_field or 'vec',
        'threshold': threshold,
        'topn': topn,
        # TODO: 由于数据集的字段可能不一致，需要重新设计LLM输入结构
        "output_fields": ["question", "answer", "file_name", "url"], 
        'csid': MAC
    }
    try:
        resp = requests.post(url, json=body, headers=REQUEST_HEADERS)
        print(resp.json())
        res = resp.json()
        return res['result']
    except Exception as e:
        raise e
    

def test_delete(domain:str, file_name:Union[List[str], str]):
    '''测试删除文件
    
    Params:
        domain(str): 数据集名称
        file_name(Union[List[str], str]): 要删除的文件名，需包括后缀，可以接受单个或批量文件名

    Return:
        dict: 任务的执行结果。
    '''
    url = urljoin(RETRIEVE_API, 'DataDeleteEngine')
    body = {
        'domain': domain,
        'file_name': file_name,
        'csid': MAC
    }
    try:
        resp = requests.post(url, json=body, headers=REQUEST_HEADERS)
        print(resp.json())
    except Exception as e:
        raise e
    

def test_precise_search(domain:str, question:str, search_field:str):
    '''测试检索出所有文件名
    
    Params:
        domain(str): 数据集名称

    Return:
        dict: 任务的执行结果。
    '''
    url = urljoin(RETRIEVE_API, 'DataPreciseSearchEngine')
    body = {
        'domain': domain, 
        'question': question, # 必传
        'search_field': search_field, # 必传
        'csid': MAC, # 必传
        'output_fields': ['question'],
    }
    try:
        resp = requests.post(url, json=body, headers=REQUEST_HEADERS)
        print(resp.json())
    except Exception as e:
        raise e
    

def test_show_collection():
    url = urljoin(RETRIEVE_API, 'VecDataEngine')
    body = {
        'csid': MAC,
    }
    try:
        resp = requests.post(url, json=body, headers=REQUEST_HEADERS)
        print(resp.json())
    except Exception as e:
        raise e
    
def test_create_collection(collections: list):
    url = urljoin(RETRIEVE_API, 'VecDataCreateEngine')
    body = {
        'csid': MAC,
        'collections': collections,
    }
    try:
        resp = requests.post(url, json=body, headers=REQUEST_HEADERS)
        print(resp.json())
    except Exception as e:
        raise e

def test_delete_collection(collections:list[str]):
    url = urljoin(RETRIEVE_API, 'VecDataDeleteEngine')
    body = {
        'csid': MAC,
        'collections': collections,
    }
    try:
        resp = requests.post(url, json=body, headers=REQUEST_HEADERS)
        print(resp.json())
    except Exception as e:
        raise e
    

if __name__ == '__main__':
    import json
    import time
    from pathlib import Path


    domain = 'test_es'

    # test_create_collection([domain])

    test_show_collection()

    # data_dir = Path('/gpfs/jincheng/csg/RAG/local_dev/业务指导书')
    # files = data_dir.rglob('*')
    # st = time.time()
    # count = 0
    # for fp in files:
    #     if fp.suffix in ['.doc', '.docx', '.pdf']:
    #         res = test_insert(str(fp), domain)
    #         count += 1
    #         res.update({'file_name': fp.name})
    #         with open('/gpfs/jincheng/csg/RAG/local_dev/insert_log.txt', 'a') as f:
    #             f.write(f'\n{json.dumps(res, indent=4, ensure_ascii=False)}\n\n')
    # et = time.time()
    # print(f'Time cost: {et - st}')
    # print(f'Docs count: {count}')

    # fp = '/gpfs/jincheng/csg/RAG/local_dev/022.TB 10101-2018_铁路工程测量规范.pdf'
    # # # fp = '/gpfs/jincheng/csg/RAG/local_dev/speak/2024/【正文】〔2024〕第4期【正文】 公司党组推进安全生产工作专题会暨2024年安全生产工作会议文件（情况通报〔2024〕第4期）-2024-02-01.docx'
    # test_insert(fp, domain, is_enhanced=False)


    query = '职工生病了，工资如何发？'
    # # query = '调度命令的形式'
    res = test_search(query, domain=domain, topn=10)
    print(len(res))
    ...


