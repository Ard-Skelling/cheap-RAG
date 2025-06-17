import requests
from uuid import uuid4


ENDPOINT = "http://127.0.0.1:8083/api/v1/query/search"


def test_query(query, domain, threshold=0.4, topk=5, output_fields=["text", "file_name", "agg_index", "url"]):
    data = {
        "csid": "test",
        "query": query,
        "domain": domain,
        "topk": topk,
        "output_fields": output_fields,
        "threshold": threshold
    }
    response = requests.post(ENDPOINT, json=data)
    return response.json()


if __name__ == "__main__":
    query = "SP600125 IL4/IL13+TNF vs DMSO IL4/IL13+TNF increase biological age"
    domain = "test"
    result = test_query(query, domain)
    texts = ''
    for item in result['result']:
        texts += f"Title: {item['file_name']}\n{item['text']}\n\n***************************************************\n\n"
    with open("result.txt", "w") as f:
        f.write(texts)