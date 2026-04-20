from ingestion.ingestion import process_multiple_pdfs
from rag.rag_pipeline import build_index, run_rag

# Load PDFs
pdf_files = ["Clinical Doc 1.pdf", "Clinical Doc 2.pdf", "Clinical Doc 3.pdf"]

# Prepare chunks
chunks = process_multiple_pdfs(pdf_files)

# 🔥 Build index ONCE
index, vectorizer = build_index(chunks)

# Query
query = "What medicines were prescribed?"

# Run RAG
result = run_rag(index, vectorizer, chunks, query)

print("\nFinal Answer:\n", result["answer"])
print("\nSources:\n", result["sources"])