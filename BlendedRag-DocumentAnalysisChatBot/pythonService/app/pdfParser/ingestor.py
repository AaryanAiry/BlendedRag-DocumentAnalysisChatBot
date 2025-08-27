import uuid
import os
from fastapi import UploadFile
from app.pdfParser.parser import extractTextFromPdf
from app.pdfParser.chunker import chunkText
from app.utils.logger import getLogger

uploadDir = "data/uploads"
logger = getLogger(__name__)

async def processPdf(file: UploadFile):
    os.makedirs(uploadDir, exist_ok=True)
    logger.info(f"Starting ingestion for: {filePath}")
    try:
        #Save the uploaded file
        docID = str(uuid.uuid4())
        filePath = os.path.join(uploadDir, f"{docID}.pdf")
        with open(filePath, "wb") as f:
            f.write(await file.read())

        #Extract text
        text, pageCount = extractTextFromPdf(filePath)
        logger.info(f"Extracted text length: {len(text)} characters")

        #Chunk text
        chunks = chunkText(text)
        logger.info(f"Generated {len(chunks)} chunks")

        return {
            "docID": docID,
            "fileName": file.filename,
            "pageCount": pageCount,
            "chunks": chunks
        }
    except Exception as e:
        logger.error(f"Ingestions failes for {filePath}: {e}")
        raise