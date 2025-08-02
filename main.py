import fitz  # PyMuPDF
import requests
import cohere
import tempfile
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# FastAPI app
app = FastAPI()

# CORS middleware (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Request body
class QARequest(BaseModel):
    documents: str
    questions: List[str]

# API Keys and Config
COHERE_API_KEY = "ba9VI3VW1sXTxyIKhOZHWPA3326tAQzHGVVQ16aI"
PINECONE_API_KEY = "pcsk_7B3Z93_8WBKxheRs5H22N8LeMJTCWzjPR1wUZKE8oUJzHDyhMot6qbZ1JrfSkKM7kcLVu7"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "pdf"
PERPLEXITY_API_KEY = "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"

# Initialize clients
co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Extract text from a remote PDF
def extract_text_from_url(pdf_url: str) -> str:
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        return " ".join(page.get_text().replace("\n", " ") for page in doc)
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

# Chunk text into manageable pieces
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Upload embeddings to Pinecone
def upload_to_pinecone(chunks: List[str]):
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    embeddings = response.embeddings
    vectors = [(f"id-{i}", vec, {"text": chunks[i]}) for i, vec in enumerate(embeddings)]
    index.upsert(vectors=vectors)

# Use Perplexity to answer the question
def ask_perplexity(query: str, context_chunks: List[str]) -> str:
    short_chunks = [" ".join(chunk.split()[:100]) for chunk in context_chunks]
    context_text = "\n\n".join(short_chunks)

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "Strictly answer the user's question in only one sentence. Do not provide explanations or extra information and don't cite your answers."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}"
            }
        ]
    }

    response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Could not fetch answer."

# Embed the query, get context from Pinecone, and call Perplexity
def ask_question(query: str, top_k: int = 1) -> str:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    query_vec = response.embeddings[0]
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    context_chunks = [match['metadata']['text'] for match in results['matches']]
    return ask_perplexity(query, context_chunks)

# Main route
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: QARequest):
    try:
        print("⏬ Downloading and parsing PDF...")
        full_text = extract_text_from_url(request.documents)

        print("📄 Chunking and embedding...")
        chunks = chunk_text(full_text)
        upload_to_pinecone(chunks)

        print("🤖 Answering questions...")
        answers = [ask_question(q) for q in request.questions]

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally (optional)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
