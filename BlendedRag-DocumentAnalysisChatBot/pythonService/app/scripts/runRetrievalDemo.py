# runRetrievalDemo.py
import requests, json, time

BASE_URL = "http://127.0.0.1:8000"

def uploadSample(pdfPath: str):
    files = {"file": open(pdfPath, "rb")}
    resp = requests.post(f"{BASE_URL}/processPdf", files=files)
    return resp.json()

def queryDoc(docId: str, query: str):
    payload = {"docId": docId, "query": query, "topK": 5, "refine": True}
    resp = requests.post(f"{BASE_URL}/api/query", json=payload)
    return resp.json()

if __name__ == "__main__":
    print("Uploading sample.pdf ...")
    up = uploadSample("sample.pdf")
    print(json.dumps(up, indent=2))
    docId = up["docId"]
    time.sleep(0.5)
    print("Querying document ...")
    q = queryDoc(docId, "What is the main purpose of the library?")
    print(json.dumps(q, indent=2))
