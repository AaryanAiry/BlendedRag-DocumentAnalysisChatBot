import os
import sys
import traceback
import asyncio

# Ensure project root (pythonService) is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ✅ Fix absolute Qwen model path
ABS_MODEL_PATH = "/home/archlinux/MyLinuxSpace/FYP/BlendedRag-DocumentAnalysisChatBot/pythonService/models/qwen2.5-3b-instruct-q5_k_m.gguf"
os.environ["MODEL_PATH"] = ABS_MODEL_PATH

from app.pdfParser import parser, chunker, ingestor
from app.embeddings import embeddingClient
from app import ragService  # note: import from the correct rag folder


# ✅ Correct PDF path inside app/data/
PDF_PATH = os.path.join(os.path.dirname(__file__), "..","pythonService", "data", "sample.pdf")
PDF_PATH = os.path.abspath(PDF_PATH)


def step1_parse_pdf():
    print("\n--- Step 1: Parsing PDF ---")
    try:
        text, pageCount = parser.extractTextFromPdf(PDF_PATH)
        print(f"✅ Extracted {len(text)} characters of text ({pageCount} pages)")
        return text
    except Exception:
        print("❌ PDF parsing failed!")
        traceback.print_exc()
        return None


def step2_chunk_text(text):
    print("\n--- Step 2: Chunking Text ---")
    try:
        chunks = chunker.chunkText(text, chunkSize=200, chunkOverlap=50)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    except Exception:
        print("❌ Chunking failed!")
        traceback.print_exc()
        return None


async def step3_ingest_pdf():
    print("\n--- Step 3: Ingesting PDF ---")
    try:
        class DummyFile:
            def __init__(self, path):
                self.filename = os.path.basename(path)
                self._path = path
            async def read(self):
                with open(self._path, "rb") as f:
                    return f.read()

        dummy_file = DummyFile(PDF_PATH)
        doc = await ingestor.processPdf(dummy_file)
        print(f"✅ Ingested docId={doc['docId']} with {len(doc['chunks'])} chunks")
        return doc
    except Exception:
        print("❌ PDF ingestion failed!")
        traceback.print_exc()
        return None


def step4_embeddings(chunks):
    print("\n--- Step 4: Generating Embeddings ---")
    try:
        client = embeddingClient.EmbeddingClient()
        embs = client.generateEmbeddings(chunks)
        print(f"✅ Generated embeddings for {len(embs)} chunks")
        return embs
    except Exception:
        print("❌ Embedding generation failed!")
        traceback.print_exc()
        return None


def step5_query(docId, query="What is Yolo sort?"):
    print("\n--- Step 5: Running Retrieval + QA ---")
    try:
        # Directly call the existing function from ragService
        response = ragService.query_document(docId, query)
        print("✅ Retrieval successful")
        print("Retrieved chunks:", response.get("retrievedChunks", []))
        print("Final Answer:", response.get("finalAnswer", ""))
    except Exception:
        print("❌ Retrieval/QA failed!")
        traceback.print_exc()


if __name__ == "__main__":
    print(f"🔍 Using model path: {ABS_MODEL_PATH}")
    print("Exists?", os.path.exists(ABS_MODEL_PATH))

    print(f"🔍 Using PDF path: {PDF_PATH}")
    print("Exists?", os.path.exists(PDF_PATH))

    text = step1_parse_pdf()
    if text:
        chunks = step2_chunk_text(text)
        doc = asyncio.run(step3_ingest_pdf())
        if doc:
            step4_embeddings([c["text"] for c in doc["chunks"]])
            step5_query(doc["docId"], "What is Yolo sort?")
