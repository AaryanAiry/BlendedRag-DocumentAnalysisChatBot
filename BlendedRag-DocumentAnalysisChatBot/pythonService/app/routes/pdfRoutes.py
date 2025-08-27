from fastapi import APIRouter, UploadFile,File,HTTPException
from app.pdfParser.ingestor import processPdf

router = APIRouter()

@router.get("/")
async def processPdfEndpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    tempPath = f"data/uploads/{file.filename}"
    with open(tempPath, "wb") as f:
        f.write(await file.read())

    # chunks = processPdf(tempPath)
    # return PDFResponse(
    #     docId=str(uuid.uuid4()),
    #     fileName=file.filename,
    #     pageCount=1,  # To be refined
    #     chunks=[Chunk(text=c) for c in chunks]
    # )
    
    result = await processPdf(file)
    return result