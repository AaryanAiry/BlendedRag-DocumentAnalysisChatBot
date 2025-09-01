
import chromadb
from chromadb.config import Settings

chromaClient = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",  
        persist_directory="./data/chroma"
    )
)
