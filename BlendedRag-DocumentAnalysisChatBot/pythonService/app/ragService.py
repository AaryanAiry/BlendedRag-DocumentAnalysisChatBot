# pythonService/app/ragService.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger
from app.rag.queryRefiner import refine_query_intelligent
from app.retrieval.blendedRetriever import BlendedRetriever
from app.embeddings.embeddingClient import embedding_fn
from app.chromaClient import get_chroma_client

logger = getLogger(__name__)

# -------------------------------
# Load Mistral-7B model (CPU safe)
# -------------------------------

MODEL_PATH = "/mnt/MyLinuxSpace/hf_cache/Mistral-7B"

if not os.path.exists(MODEL_PATH):
    raise ValueError(f"Local model path does not exist: {MODEL_PATH}")

# Force CPU to avoid GPU OOM
device = "cpu"
logger.info(f"Loading Mistral-7B from local path on device: {device}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 8-bit quantization + CPU offload for safety
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": "cpu"},   # ensure full CPU
    low_cpu_mem_usage=True,
    offload_folder="./offload"
)

logger.info("Mistral-7B loaded successfully from local path")

# -------------------------------
# Helper Functions
# -------------------------------

def build_rag_prompt(query: str, chunks: list) -> str:
    """
    Build a combined context + query prompt for Mistral.
    """
    context = "\n\n".join([chunk["text"] for chunk in chunks if chunk.get("text")])
    return f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"

def generate_answer(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Generate answer from Mistral-7B safely in small batches on CPU
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer"

# -------------------------------
# Blended RAG Service
# -------------------------------

def query_document(docId: str, user_query: str, topK: int = 5) -> dict:
    """
    Query a document using RQ + Blended Retriever + Mistral
    """
    doc = documentStore.getDocument(docId)
    if not doc:
        logger.warning(f"Document not found: {docId}")
        return {"error": "Document not found", "docId": docId}

    # Step 1: Refine Query
    rq = refine_query_intelligent(user_query)

    # Step 2: Blended Retrieval
    chroma = get_chroma_client()
    retriever = BlendedRetriever(chroma_client=chroma, embedding_fn=embedding_fn)
    collection = f"doc_{docId}"

    retrieved_docs = retriever.retrieve(
        doc_id=docId,
        collection_name=collection,
        queries=rq["variants"],
        keywords=rq["keywords"],
        top_k_final=topK
    )

    # Extract chunk texts for RAG prompt
    top_chunks = []
    for d in retrieved_docs:
        if "metadata" in d and "text" in d["metadata"]:
            top_chunks.append({"text": d["metadata"]["text"]})

    # Step 3: Generate Answer
    prompt = build_rag_prompt(user_query, top_chunks)
    answer = generate_answer(prompt)

    return {
        "docId": docId,
        "originalQuery": user_query,
        "queryRefinement": rq,
        "retrievedChunks": retrieved_docs,
        "answer": answer
    }

# -------------------------------
# Optional: Refinement function
# -------------------------------

def refine_answer(query: str, initial_answer: str, top_chunks: list) -> str:
    """
    Refine an initial answer by re-prompting Mistral with context
    """
    context = "\n\n".join([chunk["text"] for chunk in top_chunks if chunk.get("text")])
    prompt = (
        f"Given the following context:\n{context}\n\n"
        f"The previous answer to the question '{query}' was: '{initial_answer}'.\n"
        "Improve or refine this answer if necessary, correcting mistakes and adding missing information.\nAnswer:"
    )
    return generate_answer(prompt)
