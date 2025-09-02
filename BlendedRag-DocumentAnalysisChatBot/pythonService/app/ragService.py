# pythonService/app/ragService.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger

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
    offload_folder="./offload"  # optional folder for temporary offloaded tensors
)

logger.info("Mistral-7B loaded successfully from local path")

# -------------------------------
# Helper Functions
# -------------------------------

def build_rag_prompt(query: str, chunks: list) -> str:
    """
    Build a combined context + query prompt for Mistral.
    """
    context = "\n\n".join([chunk["text"] for chunk in chunks if chunk["text"]])
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    return prompt

def generate_answer(prompt: str, max_tokens: int = 512, temperature: float = 0.7, batch_size: int = 1) -> str:
    """
    Generate answer from Mistral-7B safely in small batches on CPU
    """
    try:
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate in small batches if needed
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer"

# -------------------------------
# RAG Service
# -------------------------------

def query_document(docId: str, query: str, topK: int = 5) -> dict:
    """
    Query a document from the store and generate an answer using Mistral
    """
    doc = documentStore.getDocument(docId)
    if not doc:
        logger.warning(f"Document not found: {docId}")
        return {"error": "Document not found", "docId": docId}

    # Process top-K chunks in small batches
    top_chunks = doc["chunks"][:topK]
    prompt = build_rag_prompt(query, top_chunks)
    answer = generate_answer(prompt)

    return {
        "docId": docId,
        "query": query,
        "answer": answer,
        "chunksUsed": top_chunks
    }

# -------------------------------
# Optional: Refinement function
# -------------------------------

def refine_answer(query: str, initial_answer: str, top_chunks: list) -> str:
    """
    Refine an initial answer by re-prompting Mistral with context
    """
    context = "\n\n".join([chunk["text"] for chunk in top_chunks if chunk["text"]])
    prompt = (
        f"Given the following context:\n{context}\n\n"
        f"The previous answer to the question '{query}' was: '{initial_answer}'.\n"
        "Improve or refine this answer if necessary, correcting mistakes and adding missing information.\nAnswer:"
    )
    return generate_answer(prompt)
