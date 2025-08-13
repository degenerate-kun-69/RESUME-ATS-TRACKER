import pymupdf
from langchain_community.document_loaders import PyMuPDFLoader

def input_pdf_to_text(pdf_file):
    doc = pymupdf.open(pdf_file)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])
