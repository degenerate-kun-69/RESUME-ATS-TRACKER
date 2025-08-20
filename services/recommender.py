from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from llm.langchain_setup import embedding_model, llm
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

def build_job_vector_index(job_descriptions):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = splitter.create_documents(job_descriptions)
    return FAISS.from_documents(docs, embedding=embedding_model)

def get_job_recommendations(resume_text, faiss_index, top_k=3):
    return faiss_index.similarity_search(resume_text, k=top_k)

def generate_job_recommendations(job_description, resume_text, top_k=5):
    """Generate job recommendations based on resume content"""
    try:
        recommendation_prompt = PromptTemplate.from_template("""
        Based on the following resume and job description, suggest {top_k} similar job titles/positions 
        that would be a good match for this candidate.
        
        Resume: {resume}
        
        Current Job Description: {job_description}
        
        Provide {top_k} job recommendations as a simple list, one per line:
        """)
        
        recommendation_chain = recommendation_prompt | llm | StrOutputParser()
        
        response = recommendation_chain.invoke({
            "resume": resume_text,
            "job_description": job_description,
            "top_k": top_k
        })
        
        # Parse the response into a list
        recommendations = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith(('-', '*', '•'))]
        recommendations = [rec.lstrip('123456789. ') for rec in recommendations]  # Remove numbering
        
        return recommendations[:top_k] if len(recommendations) > 0 else ["Software Engineer", "Data Analyst", "Product Manager"]
        
    except Exception as e:
        print(f"Warning: Job recommendation generation failed: {e}")
        # Return some default recommendations
        return [
            "Software Developer",
            "Data Analyst", 
            "Product Manager",
            "Business Analyst",
            "Marketing Specialist"
        ]
