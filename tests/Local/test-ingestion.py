from rag_ingestion.Local.ingestion import process_multiple_pdfs

pdf_files = ["Clinical Doc 1.pdf", "Clinical Doc 2.pdf", "Clinical Doc 3.pdf"]


chunks = process_multiple_pdfs(pdf_files)

# Print sample output
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("----------")