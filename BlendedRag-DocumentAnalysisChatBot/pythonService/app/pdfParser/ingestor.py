import uuid
import os
from fastapi import UploadFile
from app.pdfParser.parser import extractTextFromPdf
from app.pdfParser.chunker import chunkText
from app.embeddings.embeddingClient import EmbeddingClient
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger
from chromadb.config import Settings
import chromadb

# Setup Chroma client (persistent)
chromaClient = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="data/chroma"))

uploadDir = "data/uploads"
logger = getLogger(__name__)
embeddingClient = EmbeddingClient()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50  # overlap between chunks to preserve context

async def processPdf(file: UploadFile):
    os.makedirs(uploadDir, exist_ok=True)
    try:
        # Generate unique docId and file path
        docId = str(uuid.uuid4())
        filePath = os.path.join(uploadDir, f"{docId}.pdf")
        logger.info(f"Starting ingestion for: {file.filename}, saving as: {filePath}")

        # Save the uploaded file
        contents = await file.read()
        if not contents:
            raise ValueError(f"Uploaded file is empty: {file.filename}")
        with open(filePath, "wb") as f:
            f.write(contents)

        # Extract text and page count
        text, pageCount = extractTextFromPdf(filePath)
        if not text:
            raise ValueError(f"No text extracted from PDF: {file.filename}")
        logger.info(f"Extracted text length: {len(text)} characters")

        # Chunk text
        chunks = chunkText(text, chunkSize=CHUNK_SIZE, chunkOverlap=CHUNK_OVERLAP)
        logger.info(f"Generated {len(chunks)} chunks")

        # Generate embeddings
        embeddings = embeddingClient.generateEmbeddings(chunks)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")

        # Save document in memory
        documentStore.saveDocument(docId, {
            "fileName": file.filename,
            "pageCount": pageCount,
            "chunks": [{"text": chunk} for chunk in chunks],
            "embeddings": embeddings
        })

        # Save to Chroma
        collection_name = "documents"
        if collection_name not in [c.name for c in chromaClient.list_collections()]:
            collection = chromaClient.create_collection(name=collection_name)
        else:
            collection = chromaClient.get_collection(name=collection_name)

        # Prepare metadata for each chunk
        metadatas = [{"docId": docId, "chunkIndex": i, "text": chunk} for i, chunk in enumerate(chunks)]
        ids = [f"{docId}_{i}" for i in range(len(chunks))]

        collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        chromaClient.persist()
        logger.info(f"Saved {len(chunks)} chunks to Chroma for docId={docId}")

        # Return response
        return {
            "docId": docId,
            "fileName": file.filename,
            "pageCount": pageCount,
            "chunks": [{"text": chunk} for chunk in chunks]
        }

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise
