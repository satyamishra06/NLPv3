import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

CHUNKS_PATH = "data/output/chunks.json"
FAISS_INDEX_DIR = "models/faiss_index"
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
FEW_SHOT_PATH = "prompts/few_shot_examples.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
TOP_K = 3


def load_chunks(json_path: str = CHUNKS_PATH) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_system_prompt(prompt_path: str = SYSTEM_PROMPT_PATH) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_few_shot_examples(few_shot_path: str = FEW_SHOT_PATH) -> list:
    with open(few_shot_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embeddings(chunks: list, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(chunks: list, index_dir: str = FAISS_INDEX_DIR) -> None:
    os.makedirs(index_dir, exist_ok=True)

    embeddings = get_embeddings(chunks)
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built: {index.ntotal} vectors saved to {index_dir}")


def load_faiss_index(index_dir: str = FAISS_INDEX_DIR):
    index_path = os.path.join(index_dir, "index.faiss")
    chunks_path = os.path.join(index_dir, "chunks.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found at {index_path}. Run build first.")

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def retrieve(query: str, k: int = TOP_K, index_dir: str = FAISS_INDEX_DIR) -> list:
    index, chunks = load_faiss_index(index_dir)

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "score": float(distances[0][i])
            })

    return results


def build_prompt(query: str, context_chunks: list) -> list:
    system_prompt = load_system_prompt()
    few_shot = load_few_shot_examples()

    context_text = "\n\n".join([f"[Source: {c['doc_name']}]\n{c['text']}" for c in context_chunks])

    messages = [{"role": "system", "content": system_prompt}]

    for example in few_shot:
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["answer"]})

    user_message = f"""Context:
{context_text}

Question: {query}

Answer based only on the context above:"""

    messages.append({"role": "user", "content": user_message})

    return messages


def generate_answer(query: str, context_chunks: list, model: str = OLLAMA_MODEL) -> str:
    messages = build_prompt(query, context_chunks)
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def rag_query(query: str, k: int = TOP_K) -> dict:
    results = retrieve(query, k=k)
    context_chunks = [r["chunk"] for r in results]
    answer = generate_answer(query, context_chunks)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "doc_name": r["chunk"]["doc_name"],
                "chunk_id": r["chunk"]["chunk_id"],
                "score": r["score"],
                "text": r["chunk"]["text"][:200] + "..."
            }
            for r in results
        ]
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        chunks = load_chunks()
        build_faiss_index(chunks)
        sys.exit(0)

    command = sys.argv[1]

    if command == "build":
        chunks = load_chunks()
        build_faiss_index(chunks)

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python src/rag_engine.py query 'your question'")
            sys.exit(1)

        query = sys.argv[2]
        result = rag_query(query)

        print(f"\nAnswer:\n{result['answer']}")
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - {s['doc_name']} (chunk {s['chunk_id']}, score: {s['score']:.4f})")

    else:
        print(f"Unknown command: {command}")
        print("Usage: python src/rag_engine.py [build|query 'question']")
