import fitz
import numpy as np
import cohere
import requests
from pinecone import Pinecone, ServerlessSpec

# ✅ Initialize Cohere Client
co = cohere.Client("ba9VI3VW1sXTxyIKhOZHWPA3326tAQzHGVVQ16aI")  # Replace with your Cohere API key

# 1. PDF to text (cleaned)
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return " ".join(page.get_text().replace("\n", " ") for page in doc)

# 2. Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# 3. Initialize Pinecone
def init_pinecone(index_name, api_key, environment):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)  # Use existing index

# 4. Upload embeddings to Pinecone using Cohere
def upload_to_pinecone(chunks, index):
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    embeddings = response.embeddings
    pinecone_vectors = [(f"id-{i}", vec, {"text": chunks[i]}) for i, vec in enumerate(embeddings)]
    index.upsert(vectors=pinecone_vectors)

# 5. Ask question using Perplexity API (with truncated chunks)
def ask_perplexity(query, context_chunks):
    api_key = "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"
    url = "https://api.perplexity.ai/chat/completions"

    # Limit each chunk to 100 words
    short_chunks = [" ".join(chunk.split()[:100]) for chunk in context_chunks]
    context_text = "\n\n".join(short_chunks)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Strictly answer the user's question in only one sentence. Do not provide explanations or extra information and dont cite your answers"},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Error:", response.status_code, response.text)
        return None

# 6. Ask question using Cohere + Pinecone + Perplexity (reduced top_k)
def ask_question(query, index, top_k=1):
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    query_vec = response.embeddings[0]
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    context_chunks = [match['metadata']['text'] for match in results['matches']]
    return ask_perplexity(query, context_chunks)

# 🧠 Main Program
if __name__ == "__main__":
    PDF_PATH = "sample.pdf"
    PINECONE_API_KEY = "pcsk_7B3Z93_8WBKxheRs5H22N8LeMJTCWzjPR1wUZKE8oUJzHDyhMot6qbZ1JrfSkKM7kcLVu7"
    PINECONE_ENV = "us-east-1"
    INDEX_NAME = "pdf"

    print("Loading PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("Chunking and embedding...")
    chunks = chunk_text(text)

    print("Connecting to Pinecone...")
    index = init_pinecone(INDEX_NAME, PINECONE_API_KEY, PINECONE_ENV)
    upload_to_pinecone(chunks, index)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = ask_question(query, index)
        print("\nAnswer:\n", answer)
