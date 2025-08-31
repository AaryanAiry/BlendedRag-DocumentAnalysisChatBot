from typing import Dict, Any, List
import threading

class DocumentStore:
    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
_document_store: Dict[str, Dict[str, Any]] = {}

def saveDocument(docId: str, data: Dict[str, Any]) -> None:
    """
    Save a document's metadata, chunks, and embeddings in memory.
    """
    _document_store[docId] = data

def getDocument(docId: str) -> Dict[str, Any]:
    """
    Retrieve a stored document by its docId.
    Returns None if not found.
    """
    return _document_store.get(docId)
    # def saveDocument(self, docId: str, data: Dict):
    #     with self.lock:
    #         self.store[docId] = data
    
    # def getDocument(self, docId: str) -> Dict:
    #     return self.store.get(docId)
    
    def listDocuments(self) -> List[Dict]:
        # Return shallow metadata list
        with self.lock:
            return [
                {"docId": k, "fileName": v.get(fileName), "pageCount": v.get("pageCount"), "numChunks": len(v.get("chunks",[]))}
                for k,v in self.store.items()
            ]

    def deleteDocument(self, docId: str) -> bool:
        with self.lock:
            if docId in self.store:
                del self.store[docId]
                return True
            return False
        
documentStore = DocumentStore()