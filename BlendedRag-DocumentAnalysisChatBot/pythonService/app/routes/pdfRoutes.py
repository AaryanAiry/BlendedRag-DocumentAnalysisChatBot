from fastapi import APIRouter, UploadFile, File, HTTPException
from app.pdfParser.ingestor import processPdf
from app.embeddings.embeddingClient import EmbeddingClient
from app.storage.documentStore import documentStore
from app.schemas import PDFResponse, Chunk
import uuid
from app.utils.logger import getLogger
from app.config import UPLOADS_DIR
import os

router = APIRouter()
logger = getLogger(__name__)
embeddingClient = EmbeddingClient()

@router.post("/processPdf/")
async def processPdfEndpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    localFilename = f"{str(uuid.uuid4())}_{file.filename}"
    tempPath = os.path.join(UPLOADS_DIR, localFilename)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {file.filename}")

    # Save the uploaded file
    with open(tempPath, "wb") as f:
        f.write(contents)

    logger.info(f"Saved uploaded file to {tempPath}")

    # Call async processPdf with UploadFile-like object
    # Create a dummy UploadFile object pointing to the saved file
    class DummyUploadFile:
        def __init__(self, path, name):
            self.filename = name
            self._path = path
        async def read(self):
            with open(self._path, "rb") as f:
                return f.read()

    dummyFile = DummyUploadFile(tempPath, file.filename)
    uploadResult = await processPdf(dummyFile)

    return PDFResponse(
        docId=uploadResult["docId"],
        fileName=uploadResult["fileName"],
        pageCount=uploadResult["pageCount"],
        chunks=[Chunk(text=c) for c in uploadResult["chunks"]]
    )



# from fastapi import APIRouter, UploadFile,File,HTTPException
# from app.pdfParser.ingestor import processPdf
# from app.embeddings.embeddingClient import EmbeddingClient
# from app.storage.documentStore import documentStore
# from app.schemas import PDFResponse, Chunk
# import uuid
# from app.utils.logger import getLogger
# from app.config import UPLOADS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
# import os

# router = APIRouter()
# logger = getLogger(__name__)
# embeddingClient = EmbeddingClient()

# @router.post("/")
# async def processPdfEndpoint(file: UploadFile = File(...)):
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     os.makedirs(UPLOADS_DIR, exist_ok=True)
#     localFilename = f"{str(uuid.uuid4())}_{file.filename}"
#     tempPath = os.path.join(UPLOADS_DIR, localFilename)
#     with open(tempPath, "wb") as f:
#         f.write(await file.read())

#     logger.info(f"Saved uploaded file to {tempPath}")
#     # ingestPdf should return list of chunk texts or list of dicts; adjust accordingly
#     # chunks = processPdf(tempPath, chunkSize=CHUNK_SIZE, chunkOverlap=CHUNK_OVERLAP)
#     # chunks assumed to be list of dicts like {"text": "..."}
#     uploadResult = await processPdf(file)  # async call, UploadFile passed directly
#     chunks = [{"text": c} for c in uploadResult["chunks"]]  # wrap text as dict for PDFResponse

#     texts = [c["text"] for c in chunks]

#     embeddingArray = embeddingClient.getEmbeddings(texts)  # numpy array
#     # convert to lists for JSON-serializable storage
#     embeddingsAsLists = embeddingArray.tolist()

#     docId = str(uuid.uuid4())
#     documentStore.saveDocument(docId, {
#         "fileName": file.filename,
#         "pageCount": 1,  # update later if parser returns page count
#         "chunks": chunks,
#         "embeddings": embeddingsAsLists
#     })

#     logger.info(f"Stored document {docId} with {len(chunks)} chunks")
#     return PDFResponse(
#         docId=docId,
#         fileName=file.filename,
#         pageCount=1,
#         chunks=[Chunk(text=c["text"]) for c in chunks]
#     )

#     # deprecated:
#     # tempPath = f"data/uploads/{file.filename}"
#     # with open(tempPath, "wb") as f:
#     #     f.write(await file.read())

#     # embeddings = embeddingClient.getEmbeddings(chunks)
    
    
#     # chunks = processPdf(tempPath)
#     # return PDFResponse(
#     #     docId=str(uuid.uuid4()),
#     #     fileName=file.filename,
#     #     pageCount=1,  # To be refined
#     #     chunks=[Chunk(text=c) for c in chunks]
#     # )
    
#     # result = await processPdf(file)
#     # return result