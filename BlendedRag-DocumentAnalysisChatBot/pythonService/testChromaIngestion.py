# testChromaIngestion.py
import asyncio
from app.pdfParser.ingestor import processPdf
from app.chromaClient import collection
from app.embeddings.embeddingClient import EmbeddingClient
from fastapi import UploadFile

embeddingClient = EmbeddingClient()

class DummyUploadFile:
    """Simulate UploadFile for local PDF"""
    def __init__(self, path, name):
        self.filename = name
        self._path = path

    async def read(self):
        with open(self._path, "rb") as f:
            return f.read()

async def test():
    # Use a small PDF for testing
    testPdfPath = "data/sample.pdf"  # make sure this exists
    dummyFile = DummyUploadFile(testPdfPath, "sample.pdf")

    # Process PDF
    result = await processPdf(dummyFile)
    docId = result["docId"]
    chunks = [c["text"] for c in result["chunks"]]

    # Generate embeddings
    embeddings = embeddingClient.generateEmbeddings(chunks)

    # Add to Chroma collection
    collection.add(
        ids=[f"{docId}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"docId": docId, "chunkIndex": i, "text": chunks[i]} for i in range(len(chunks))]
    )

    print(f"Stored {len(chunks)} chunks in Chroma for docId={docId}")

if __name__ == "__main__":
    asyncio.run(test())
