import re
import pickle
from pathlib import Path

import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
INDEX_DIR = Path(__file__).resolve().parent / "index"
TOP_K = 4

_model = SentenceTransformer(EMBED_MODEL)
_index = faiss.read_index(str(INDEX_DIR / "thesis.index"))
with open(INDEX_DIR / "chunks.pkl", "rb") as f:
    _chunks = pickle.load(f)
_client = Groq()


def retrieve(query, k=TOP_K):
    query_vector = _model.encode([query], normalize_embeddings=True)
    scores, indices = _index.search(query_vector, k)
    return [_chunks[i] for i in indices[0]]


def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in chunks
    )
    return f"""Answer the question using only the context below. You may combine and compare
information from different passages; an answer assembled from several passages
is still grounded in the context. Only say you don't have that information if
the context genuinely lacks what is needed. Cite the page number in round brackets, like (page 12). Never write the file name.
Context:
{context}

Question: {query}

Answer:"""

_CITE = re.compile(r"[\[(]\s*[^\])]*?pages?\s*(\d+(?:\s*[\u2013,-]\s*\d+)*)\s*[\])]", re.IGNORECASE)


def clean_citations(text):
    text = (text or "").replace("\u3010", "[").replace("\u3011", "]")
    text = _CITE.sub(r"(page \1)", text)
    return re.sub(r"\[\s*p\.?\s*(\d+)\s*\]", r"(page \1)", text)

def answer(query, k =TOP_K):
    chunks = retrieve(query, k)
    prompt = build_prompt(query, chunks)
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = clean_citations(response.choices[0].message.content)
    sources = [{"source": c["source"], "page": c["page"]} for c in chunks]
    return text, sources


if __name__ == "__main__":
    question = "How does the system guide users during an emergency evacuation?"
    text, sources = answer(question)
    print(f"Q: {question}\n")
    print(text)
    print("\nSources:")
    for s in sources:
        print(f"- {s['source']} (page {s['page']})")
