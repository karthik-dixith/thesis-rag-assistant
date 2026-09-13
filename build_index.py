import pickle 
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from ingest import load_pages, chunk_pages, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = Path(__file__).resolve().parent / "index"

def build():
    pages = load_pages(DATA_DIR)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer (MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "thesis.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


    print(f"Embedded {len(chunks)} chunks into {dimension}-dim vectors")
    print(f"Saved index and metadata to {INDEX_DIR}")
    return model, index, chunks


def smoke_test(model, index, chunks, query, k=3):
    query_vector = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_vector, k)
    print(f"\nSmoke test query: {query!r}")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        hit = chunks[idx]
        print(f"\n#{rank} score={score:.3f} (page {hit['page']})")
        print(hit["text"][:200])



if __name__ == "__main__":
    model, index, chunks = build()
    smoke_test(model, index, chunks, "how does the system guide users during evacuation?")
        