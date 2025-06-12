import os
import base64
import requests
import uuid
import time

def test_insert_pdf():
    # 测试服务器地址
    base_url = "http://localhost:8002"
    
    # 读取PDF文件
    pdf_path = "tests/data/DSAC-v2.pdf"
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    # 将PDF内容转换为base64
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    
    # 准备请求数据
    data = {
        "file_bs64": pdf_base64,
        "domain": "paper_test",
        "file_name": "DSAC-v2.pdf",
        "csid": str(uuid.uuid4()),
        "ocr_api": os.getenv('OCR_API'),
        "emb_api": os.getenv('EMB_API')
    }
    
    # 发送请求
    response = requests.post(
        f"{base_url}/api/v1/insert/insert_pdf",
        json=data
    )
    
    # 打印响应
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return data["domain"], data["csid"]

def test_search(domain: str, csid: str):
    base_url = "http://localhost:8002"
    
    # 准备查询请求数据
    data = {
        "query": "reward function",  # 示例查询词
        "domain": domain,
        "threshold": 0.01,
        "topk": 5,
        "output_fields": ["text", "file_name", "agg_index", "url"],
        "csid": csid
    }
    
    # 发送查询请求
    response = requests.post(
        f"{base_url}/api/v1/query/search",
        json=data
    )
    
    print("\n=== 查询结果 ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

def test_delete_domain(domain: str, csid: str):
    base_url = "http://localhost:8002"
    
    # 准备删除请求数据
    data = {
        "collections": domain,
        "csid": csid
    }
    
    # 发送删除请求
    response = requests.post(
        f"{base_url}/api/v1/maintain/delete_domain",
        json=data
    )
    
    print("\n=== 删除结果 ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def run_all_tests():
    print("=== 开始测试 ===")
    
    # 1. 插入PDF
    print("\n=== 插入PDF ===")
    domain, csid = test_insert_pdf()
    
    # 等待一段时间确保插入完成
    print("\n等待5秒确保插入完成...")
    time.sleep(5)
    
    # 2. 查询
    test_search(domain, csid)
    
    # 3. 删除数据库
    test_delete_domain(domain, csid)

if __name__ == "__main__":
    run_all_tests()
