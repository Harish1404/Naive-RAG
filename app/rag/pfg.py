from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def load_pdf(path: str):

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)

    words = text.split()

    chunks = []

    chunk_size, overlap = 300, 50

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_word = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_word))

    model_name = "all-MiniLM-L6-v2"
    
    model = SentenceTransformer(model_name)

    embed = model.encode(chunks).tolist()

    return embed

print(load_pdf("uploads\my_resume.pdf"))
