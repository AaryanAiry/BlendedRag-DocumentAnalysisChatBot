import uuid
import os
from fastapi import UploadFile
from app.pdfParser.parser import extractTextFromPdf
from app.pdfParser.chunker import chunkText

uploadDir = "data/uploads"

async def processPdf(file: UploadFile):
    os.makedirs(uploadDir, exist_ok=True)

    #Save the uploaded file
    docID = str(uuid.uuid4())
    filePath = os.path.join(uploadDir, f"{docID}.pdf")
    with open(filePath, "wb") as f:
        f.write(await file.read())

    #Extract text
    text, pageCount = extractTextFromPdf(filePath)

    #Chunk text
    chunks = chunkText(text)

    return {
        "docID": docID,
        "fileName": file.filename,
        "pageCount": pageCount,
        "chunks": chunks
    }