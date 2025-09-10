# app/ui/streamlitApp.py
import streamlit as st
import requests
import chromadb
from typing import Dict, List

# FastAPI endpoint URLs
API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/processPdf"
DOC_LIST_ENDPOINT = f"{API_BASE_URL}/DocRoute/api/documents"
DELETE_DOC_ENDPOINT = f"{API_BASE_URL}/DocRoute/api/documents/{{docId}}"
RAG_ENDPOINT = f"{API_BASE_URL}/rag/api/ask"
QUERY_ENDPOINT = f"{API_BASE_URL}/queryPdf/api/query"

# Chroma client for debug view
chroma_client = chromadb.PersistentClient(path="./data/chroma")

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []

# Helper function to call FastAPI
def call_api(endpoint: str, method: str = "GET", json_data: Dict = None, files: Dict = None):
    try:
        if method == "POST":
            response = requests.post(endpoint, json=json_data, files=files, timeout=30)
        elif method == "DELETE":
            response = requests.delete(endpoint, timeout=30)
        else:
            response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API call failed: {e}")
        return None

# Sidebar: PDF Upload and Document List
st.sidebar.title("Document Management")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
if uploaded_file:
    with st.status("Processing PDF...", expanded=True) as status:
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        result = call_api(UPLOAD_ENDPOINT, method="POST", files=files)
        if result:
            status.update(label=f"Uploaded {result['fileName']} (ID: {result['docId']})", state="complete")
            st.sidebar.success(f"Uploaded {result['fileName']} (ID: {result['docId']})")
        else:
            status.update(label="Failed to upload PDF", state="error")

# Document List with Refresh
st.sidebar.subheader("Uploaded Documents")
if st.sidebar.button("Refresh List", key="refresh_docs"):
    st.rerun()
docs = call_api(DOC_LIST_ENDPOINT)
if docs and docs.get("documents"):
    for doc in docs["documents"]:
        file_name = doc.get("fileName", "unknown.pdf")
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"{file_name} (Chunks: {doc.get('numChunks', 0)})")
        if col2.button("Delete", key=f"delete_{doc['docId']}"):
            result = call_api(DELETE_DOC_ENDPOINT.format(docId=doc['docId']), method="DELETE")
            if result:
                st.sidebar.success(f"Deleted {doc['fileName']}")
                st.rerun()
else:
    st.sidebar.write("No documents uploaded.")
    if docs:
        st.sidebar.write(f"Debug: API response - {docs}")  # Log response for debugging

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Query PDF", "Ask RAG", "Chroma Debug"])

# Tab 1: Query PDF
with tab1:
    st.header("Query PDF")
    doc_options = {doc["fileName"]: doc["docId"] for doc in docs["documents"]} if docs and docs.get("documents") else {}
    selected_doc = st.selectbox("Select Document", options=list(doc_options.keys()), key="query_doc")
    query = st.text_input("Enter your query", key="query_input")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5, key="query_topk")
    refine = st.checkbox("Refine Query", value=True, key="query_refine")
    if st.button("Submit Query", key="query_submit"):
        if query and selected_doc:
            with st.spinner("Querying..."):
                doc_id = doc_options[selected_doc]
                payload = {"docId": doc_id, "query": query, "topK": top_k, "refine": refine}
                result = call_api(QUERY_ENDPOINT, method="POST", json_data=payload)
                if result:
                    st.session_state.query_history.append({
                        "query": query,
                        "doc": selected_doc,
                        "results": result["results"],
                        "mergedBlocks": result["mergedBlocks"],
                        "refinedQueries": result["refinedQueries"]
                    })
                    st.rerun()
    for chat in st.session_state.query_history:
        with st.chat_message("user"):
            st.write(f"Query: {chat['query']} (Doc: {chat['doc']})")
        with st.chat_message("assistant"):
            st.write("**Results**:")
            for res in chat["results"][:3]:
                st.write(f"- Chunk {res['chunkIndex']}: {res['snippet'][:100]}... (Score: {res['score']:.4f})")
            st.write("**Merged Context**:")
            st.write(chat["mergedBlocks"][0][:200] + "..." if chat["mergedBlocks"] else "No context")

# Tab 2: Ask RAG
with tab2:
    st.header("Ask RAG")
    doc_options = {doc["fileName"]: doc["docId"] for doc in docs["documents"]} if docs and docs.get("documents") else {}
    selected_doc = st.selectbox("Select Document", options=list(doc_options.keys()), key="rag_doc")
    query = st.text_input("Enter your query", key="rag_input")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5, key="rag_topk")
    if st.button("Submit RAG Query", key="rag_submit"):
        if query and selected_doc:
            with st.spinner("Querying RAG..."):
                doc_id = doc_options[selected_doc]
                payload = {"docId": doc_id, "query": query, "topK": top_k}
                result = call_api(RAG_ENDPOINT, method="POST", json_data=payload)
                if result and "error" not in result:
                    st.session_state.rag_history.append({
                        "query": query,
                        "doc": selected_doc,
                        "finalAnswer": result["finalAnswer"],
                        "retrievedChunks": result["retrievedChunks"]
                    })
                    st.rerun()
                elif result:
                    st.error(result["error"])
    for chat in st.session_state.rag_history:
        with st.chat_message("user"):
            st.write(f"Query: {chat['query']} (Doc: {chat['doc']})")
        with st.chat_message("assistant"):
            st.write(f"**Answer**: {chat['finalAnswer']}")
            with st.expander("Show Retrieved Chunks"):
                for chunk in chat["retrievedChunks"][:3]:
                    text = chunk["chunk"] if isinstance(chunk["chunk"], str) else chunk["chunk"].get("text", "")
                    st.write(f"- Chunk: {text[:100]}... (Score: {chunk['score']:.4f})")

# Tab 3: Chroma Debug
with tab3:
    st.header("Chroma Debug")
    collections = chroma_client.list_collections()
    st.subheader("Collections")
    for coll in collections:
        st.write(f"- {coll.name} (Docs: {coll.count()})")
    
    collection = chroma_client.get_collection("documents")
    if st.button("Peek at Documents", key="peek_docs"):
        results = collection.get(include=["metadatas", "documents"])
        if results["metadatas"]:
            st.subheader("Sample Documents")
            for i, (meta, doc) in enumerate(zip(results["metadatas"][:5], results["documents"][:5])):
                with st.expander(f"Doc {meta['docId']} (Chunk {meta['chunkIndex']})"):
                    st.write(f"**File**: {meta.get('fileName', 'N/A')}")
                    st.write(f"**Page Count**: {meta.get('pageCount', 'N/A')}")
                    st.write(f"**Text**: {(doc[:200] + '...') if doc else 'No text available'}")
    
    debug_query = st.text_input("Debug Query (searches all docs)", key="debug_query")
    if st.button("Run Debug Query", key="debug_submit"):
        if debug_query:
            results = collection.query(query_texts=[debug_query], n_results=5, include=["metadatas", "documents", "distances"])
            if results["documents"]:
                st.subheader("Query Results")
                for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
                    with st.expander(f"Result {i+1} (Score: {dist:.4f})"):
                        st.write(f"**Doc ID**: {meta['docId']}")
                        st.write(f"**Chunk Index**: {meta['chunkIndex']}")
                        st.write(f"**Text**: {(doc[:200] + '...') if doc else 'No text available'}")