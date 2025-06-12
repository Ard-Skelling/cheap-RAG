import os
import base64
import requests
import uuid
import time

def test_insert_pdf():
    base_url = "http://localhost:8002"
    pdf_path = "tests/data/DSAC-v2.pdf"
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    data = {
        "file_bs64": pdf_base64,
        "domain": "paper_test",
        "file_name": "DSAC-v2.pdf",
        "csid": str(uuid.uuid4()),
        "ocr_api": os.getenv('OCR_API'),
        "emb_api": os.getenv('EMB_API')
    }
    response = requests.post(
        f"{base_url}/api/v1/insert/insert_pdf",
        json=data
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return data["domain"], data["csid"]

def test_search(domain: str, csid: str):
    base_url = "http://localhost:8002"
    data = {
        "query": "reward function", 
        "domain": domain,
        "threshold": 0.01,
        "topk": 5,
        "output_fields": ["text", "file_name", "agg_index", "url"],
        "csid": csid
    }

    response = requests.post(
        f"{base_url}/api/v1/query/search",
        json=data
    )
    
    print("\n=== Query Result ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

def test_delete_domain(domain: str, csid: str):
    base_url = "http://localhost:8002"
    
    data = {
        "collections": domain,
        "csid": csid
    }
    
    response = requests.post(
        f"{base_url}/api/v1/maintain/delete_domain",
        json=data
    )
    
    print("\n=== Delete Result ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def run_all_tests():
    print("=== Start Test ===")
    
    print("\n=== Insert PDF ===")
    domain, csid = test_insert_pdf()
    
    print("\nWait 5 seconds to ensure insertion is complete...")
    time.sleep(5)
    
    test_search(domain, csid)
    
    test_delete_domain(domain, csid)

if __name__ == "__main__":
    run_all_tests()
