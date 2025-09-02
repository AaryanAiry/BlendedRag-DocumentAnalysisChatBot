# app/routes/pdfRoutes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.pdfParser.ingestor import processPdf
from app.embeddings.embeddingClient import EmbeddingClient
from app.schemas import PDFResponse, Chunk
import uuid
from app.utils.logger import getLogger
from app.config import UPLOADS_DIR
import os
from app.chromaClient import collection  # Only collection is needed

router = APIRouter()
logger = getLogger(__name__)
embeddingClient = EmbeddingClient()

@router.post("/processPdf/")
async def processPdfEndpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    localFilename = f"{uuid.uuid4()}_{file.filename}"
    tempPath = os.path.join(UPLOADS_DIR, localFilename)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {file.filename}")

    with open(tempPath, "wb") as f:
        f.write(contents)

    logger.info(f"Saved uploaded file to {tempPath}")

    class DummyUploadFile:
        def __init__(self, path, name):
            self.filename = name
            self._path = path

        async def read(self):
            with open(self._path, "rb") as f:
                return f.read()

    dummyFile = DummyUploadFile(tempPath, file.filename)
    uploadResult = await processPdf(dummyFile)

    docId = uploadResult["docId"]
    chunks = [chunk["text"] for chunk in uploadResult["chunks"]]

    # Generate embeddings
    embeddings = embeddingClient.generateEmbeddings(chunks)

    # Add to Chroma collection
    collection.add(
        ids=[f"{docId}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{
            "docId": docId,
            "chunkIndex": i,
            "text": chunks[i]
        } for i in range(len(chunks))]
    )

    logger.info(f"Stored {len(chunks)} chunks with embeddings in ChromaDB for docId: {docId}")

    return PDFResponse(
        docId=docId,
        fileName=uploadResult["fileName"],
        pageCount=uploadResult["pageCount"],
        chunks=[Chunk(text=c) for c in chunks]
    )
