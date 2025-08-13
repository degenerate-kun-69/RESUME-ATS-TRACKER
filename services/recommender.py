from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm.langchain_setup import embedding_model

def build_job_vector_index(job_descriptions):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = splitter.create_documents(job_descriptions)
    return FAISS.from_documents(docs, embedding=embedding_model)

def get_job_recommendations(resume_text, faiss_index, top_k=3):
    return faiss_index.similarity_search(resume_text, k=top_k)

def generate_job_recommendations(job_description, resume_text, top_k=5):
    #placeholder for gemini api call fix
    ...
