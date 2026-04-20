from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    print(f"Number of pages in {file_path}: {len(reader.pages)}")
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        print(f"Extracted text from page: {extracted[:100]}...")  # Print first 100 characters of extracted text    
        if extracted: 
            text += extracted + "\n"

    return text


def chunk_text_with_metadata(text, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)


    return [{"text": chunk, "source": source} for chunk in chunks]



def process_multiple_pdfs(file_paths):
    all_chunks = []

    for path in file_paths:
        text = extract_text_from_pdf(path)
        chunks = chunk_text_with_metadata(text, path)
        all_chunks.extend(chunks)

    return all_chunks