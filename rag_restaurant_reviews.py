"""
rag_restaurant_reviews.py

Simple local RAG system for restaurant reviews:
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: Chroma (persistent, local, new API ≥0.5.0)
- Reranker: TinyLlama via Ollama
- Generator: Gemma 2B via Ollama

Requirements:
    pip install chromadb sentence-transformers pandas requests numpy

Make sure Ollama is running and models are pulled:
    ollama pull tinyllama
    ollama pull gemma:2b

Run:
    python rag_restaurant_reviews.py
"""

import os
import numpy as np
import pandas as pd
import requests
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ------------- Config -------------
CSV_PATH = "realistic_restaurant_reviews.csv"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "restaurant_reviews"
EMBED_MODEL = "all-MiniLM-L6-v2"
RETRIEVER_K = 7  # Get a few extra for reranking

OLLAMA_API = "http://localhost:11434/api/generate"
RETRIEVER_MODEL = "tinyllama"
GENERATOR_MODEL = "gemma:2b"
# -----------------------------------

def call_ollama(model: str, prompt: str, stream: bool = False, timeout: int = 180):
    """Call Ollama /api/generate endpoint and return the generated text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=timeout)
        r.raise_for_status()
        response = r.json()
        return response.get("response", str(response))
    except Exception as e:
        return f"[Ollama error: {e}]"


# --------- Embedding with SBERT ----------
class SBertEmbedder:
    def __init__(self, model_name=EMBED_MODEL):
        print("Loading SBERT model (this may take a moment)...")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype("float32").tolist()


# --------- Chroma (new API) ----------
def get_chroma_client(persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    # New persistent client (Chroma ≥0.5.0)
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def build_or_load_collection(client, collection_name: str, csv_path: str, embedder: SBertEmbedder):
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Loaded existing collection: {collection_name} ({collection.count()} documents)")
        return collection
    except Exception:
        print("No existing collection found. Building from CSV...")

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        texts = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            title = row.get("Title", "")
            review = row.get("Review", "")
            content = f"{title}\n\n{review}".strip()

            texts.append(content)
            metadatas.append({
                "rating": str(row.get("Rating", "")),
                "date": str(row.get("Date", "")),
                "source": "review"
            })
            ids.append(f"doc_{idx}")

        print(f"Embedding {len(texts)} reviews...")
        embeddings = embedder.embed(texts)

        collection = client.create_collection(name=collection_name)
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Created collection '{collection_name}' with {len(texts)} documents.")
        return collection


# --------- Reranker with TinyLlama ----------
def rerank_with_tinyllama(question: str, chunks: list):
    if len(chunks) <= 3:
        return "\n\n---\n\n".join(chunks)

    prompt = f"""You are a precise relevance ranker.
Question: {question}

Below are several review chunks. Select the TOP 3 most relevant ones.
Return ONLY the selected chunks, separated by exactly "---" (three dashes).
Do not add numbers, explanations, or extra text.

"""
    for i, chunk in enumerate(chunks, 1):
        prompt += f"CHUNK {i}:\n{chunk}\n\n"

    response = call_ollama(RETRIEVER_MODEL, prompt, timeout=90)

    if "---" in response:
        selected = [part.strip() for part in response.split("---") if part.strip()]
        if selected:
            return "\n\n---\n\n".join(selected[:3])

    # Fallback: return top 3 original
    return "\n\n---\n\n".join(chunks[:3])


# --------- Answer Generator with Gemma ----------
GEN_PROMPT = """You are a helpful and honest assistant.
Use only the provided context to answer the question.
If the context doesn't have enough information, say "I don't have enough information to answer confidently."

Context:
{context}

Question: {question}

Answer in a clear, friendly, and concise way. Include ratings or dates if relevant.
"""

def generate_answer(context: str, question: str):
    prompt = GEN_PROMPT.format(context=context, question=question)
    return call_ollama(GENERATOR_MODEL, prompt, timeout=240)


# --------- Main Loop ----------
def main():
    print("Restaurant Review RAG System (Chroma + Ollama)")
    print("Make sure Ollama is running!\n")

    if not Path(CSV_PATH).exists():
        print(f"Error: {CSV_PATH} not found!")
        print("Please place your CSV file in the same folder.")
        return

    embedder = SBertEmbedder()
    client = get_chroma_client(CHROMA_DIR)
    collection = build_or_load_collection(client, COLLECTION_NAME, CSV_PATH, embedder)

    print("\nReady! Ask questions about the restaurant reviews.")
    print("Type 'q' or 'quit' to exit.\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        print("Searching reviews...")

        # Embed query
        query_emb = embedder.embed(query)[0]

        # Retrieve top K
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=RETRIEVER_K,
            include=["documents", "metadatas", "distances"]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            print("No matching reviews found.")
            continue

        # Optional: rerank with TinyLlama
        print("Reranking results with TinyLlama...")
        reranked_context = rerank_with_tinyllama(query, docs)

        # Generate final answer
        print("Generating answer with Gemma...")
        answer = generate_answer(reranked_context, query)

        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(answer.strip())
        print("="*60 + "\n")


if __name__ == "__main__":
    main()