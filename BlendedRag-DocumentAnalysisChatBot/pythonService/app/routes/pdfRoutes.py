from fastapi import APIRouter, UploadFile,File,HTTPException
from app.pdfParser.ingestor import processPdf

router = APIRouter()

@router.get("/")
async def processPdfEndpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    result = await processPdf(file)
    return result