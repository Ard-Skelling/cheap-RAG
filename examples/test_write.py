import base64
import requests
import uuid
import time
import json
import itertools
import threading
from os import getenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


RAG_ENDPOINT = "http://localhost:8002"
CSID = "1234567890"


class RoundRobin:
    def __init__(self, hosts: list[str]):
        if not hosts:
            raise ValueError("Hosts list cannot be empty.")
        # Create a cycle iterator
        self._hosts_cycle = itertools.cycle(hosts)
        self._lock = threading.Lock() # Protect the access to the cycle iterator

    def get_next_host(self) -> str:
        """
        Get the next host in the round-robin sequence.
        This method is thread-safe.
        """
        with self._lock:
            # next() will get the next element from the cycle iterator
            return next(self._hosts_cycle)


def test_insert_pdf(domain: str, pdf_path: str, ocr_endpoint: str, emb_endpoint: str):
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    data = {
        "file_bs64": pdf_base64,
        "domain": domain,
        "file_name": Path(pdf_path).name,
        "csid": CSID,
        "ocr_api": f'{ocr_endpoint}/v1/ocr',
        "emb_api": f'{emb_endpoint}/v1/embedding'
    }
    response = requests.post(
        f"{RAG_ENDPOINT}/api/v1/insert/insert_pdf",
        json=data
    )
    return response.json()


def test_search(domain: str, query: str):
    data = {
        "query": query, 
        "domain": domain,
        "threshold": 0.01,
        "topk": 5,
        "output_fields": ["text", "file_name", "agg_index", "url"],
        "csid": CSID
    }

    response = requests.post(
        f"{RAG_ENDPOINT}/api/v1/query/search",
        json=data
    )
    return response.json()

def test_delete_paper(domain: str, file_name: str):
    base_url = f"{RAG_ENDPOINT}/api/v1/maintain/delete_paper"
    data = {
        "csid": CSID,
        "domain": domain,
        "file_name": file_name
    }
    response = requests.post(base_url, json=data)
    return response.json()

def test_list_paper(domain: str):
    base_url = f"{RAG_ENDPOINT}/api/v1/maintain/list_paper"
    data = {
        "csid": CSID,
        "domain": domain
    }
    response = requests.post(base_url, json=data)
    return response.json()

def bulk_insert(pdfs: list[str], available_ocr_endpoints: list[str], available_emb_endpoints: list[str], log_file: str):
    ocr_scheduler = RoundRobin(available_ocr_endpoints)
    emb_scheduler = RoundRobin(available_emb_endpoints)

    def insert_pdf(pdf_path: str, ocr_scheduler: RoundRobin, emb_scheduler: RoundRobin):
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
        ocr_endpoint = ocr_scheduler.get_next_host()
        emb_endpoint = emb_scheduler.get_next_host()
        data = {
            "file_bs64": pdf_base64,
            "domain": "paper_test",
            "file_name": Path(pdf_path).name,
            "csid": CSID,
            "ocr_api": f'{ocr_endpoint}/v1/ocr',
            "emb_api": f'{emb_endpoint}/v1/embedding'
        }
        response = requests.post(
            f"{RAG_ENDPOINT}/api/v1/insert/insert_pdf",
            json=data
        )
        return response.json()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(insert_pdf, pdf, ocr_scheduler, emb_scheduler) for pdf in pdfs]
        for future in as_completed(futures):
            result = future.result()
            with open(log_file, "a") as f:
                f.write(f"{json.dumps(result, indent=4, ensure_ascii=False)}\n\n")

if __name__ == "__main__":
    ocr_endpoint = "http://127.0.0.1:34730"
    emb_endpoint = "http://127.0.0.1:34960"
    domain = "paper_test"
    base_dir = Path(__file__).parent.parent
    pdf_path = str(base_dir / "examples" / "data" / "DSAC-v2.pdf")
    query = "objective function"
    result = test_insert_pdf(domain, pdf_path, ocr_endpoint, emb_endpoint)
    print(result)
    time.sleep(5)
    result = test_list_paper(domain)
    print(result)
    result = test_search(domain, query)
    print(result)
    result = test_delete_paper(domain, "DSAC-v2.pdf")
    print(result)
    result = test_search(domain, query)
    print(result)
    with open(base_dir / "dev" / "recall_3000.json", "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    ...
