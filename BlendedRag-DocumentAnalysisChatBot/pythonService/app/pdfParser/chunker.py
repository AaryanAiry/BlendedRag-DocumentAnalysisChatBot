def chunkText(text: str, chunkSize: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size,len(words))
        chunk = "".join(words[start:end])
        chunks.append({"text": chunk})
        start+= chunkSize - overlap
    return chunks