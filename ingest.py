from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent / "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_pages(data_dir):
    pages = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"source": pdf_path.name, "page": page_number, "text": text})
    return pages

def chunk_pages(pages, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({"source": page["source"], "page":page["page"], "text":piece})
    return chunks

def main():
    pages = load_pages(DATA_DIR)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Loaded {len(pages)} pages")
    print(f"Created {len(chunks)} chunks")
    if chunks:
        sample = chunks[len(chunks) //2]
        print(f"\nSample chunk (source={sample['source']}, page={sample['page']}):")
        print(sample["text"][:500])



if __name__ == "__main__":
    main()