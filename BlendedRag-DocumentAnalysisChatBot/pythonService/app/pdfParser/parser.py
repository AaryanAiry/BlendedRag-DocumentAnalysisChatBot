import fitz

def extractTextFromPdf(filePath: str):
    text = []
    with fitz.open(filePath) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "/n".join(text), len(doc)