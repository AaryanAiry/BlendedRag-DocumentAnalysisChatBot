from fastapi import FastAPI
from app.routes import healthRoutes, pdfRoutes

app = FastAPI(title="Document AI Engine")

#Registering routes
app.include_router(healthRoutes.router, prefix="/health",tags=["Health"])
app.include_router(pdfRoutes.router, prefix="/processPdf", tags=["PDF Processing"])

@app.get("/")
def root():
    return {"message" : "Document AI Engine is running"}