from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from config import GOOGLE_API_KEY
import google.generativeai as genai
from llm.tools import confidence_score_calculation_tool, hiring_decision_tool
from langchain_community.vectorstores import FAISS
import os
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.15)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize vector store - create empty one if it doesn't exist
vector_store_path = "vector_store"
if os.path.exists(vector_store_path):
    try:
        vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Warning: Could not load existing vector store: {e}")
        print("Creating empty vector store...")
        # Create empty vector store with dummy document
        from langchain.schema import Document
        dummy_doc = Document(page_content="dummy", metadata={})
        vector_store = FAISS.from_documents([dummy_doc], embedding_model)
        vector_store.save_local(vector_store_path)
else:
    print("Vector store not found. Creating empty vector store...")
    # Create empty vector store with dummy document
    from langchain.schema import Document
    dummy_doc = Document(page_content="dummy", metadata={})
    vector_store = FAISS.from_documents([dummy_doc], embedding_model)
    vector_store.save_local(vector_store_path)

classification_prompt = PromptTemplate.from_template("""
You are a skilled recruiter that is highly experienced in evaluating candidates for various positions,
with a deep understanding of the skills and qualifications required for different roles.
Your task is to analyze the candidate's resume and provide a detailed evaluation of their suitability for the position based on the provided job description.
You must consider the job market to be highly competitive.

Resume:
{resume}

Job Description:
{job_description}

Your task is to:
1. Highlight any missing keywords and skills
2. Provide a concise profile summary

Respond strictly in this JSON format:
{{
  "Missing Keywords and Skills": ["skill1", "skill2", "..."],
  "Profile Summary": "detailed summary..."
}}
""")

classifier_chain = classification_prompt | llm | StrOutputParser()

def evaluate_resume(resume:str, job_description:str, vector_store_param=None)-> dict:
    """
    Evaluate a resume against a job description using FAISS similarity for deterministic scoring
    and an LLM for qualitative analysis.

    Args:
        resume (str): The candidate's resume.
        job_description (str): The job description for the position.
        vector_store_param: The vector store for similarity calculations.

    Returns:
        dict: Evaluation results including job description match, missing skills, confidence score, profile summary, and hiring decision.
    """
    # Use global vector_store if none provided
    vs = vector_store_param if vector_store_param is not None else vector_store
        
    # Calculate FAISS similarity
    try:
        faiss_similarity = vs.similarity_search_with_score(job_description, k=1)
        _, distance = faiss_similarity[0]
        similarity_percentage = max(0, min(100, 100 - (distance * 100)))
    except Exception as e:
        print(f"Warning: FAISS similarity calculation failed: {e}")
        similarity_percentage = 50.0  # Default fallback

    # confidence score calculation
    confidence_score = confidence_score_calculation_tool.invoke(similarity_percentage)

    #hiring decision
    decision = hiring_decision_tool.invoke(confidence_score)
    
    # Get LLM response
    try:
        llm_response_text = classifier_chain.invoke({
            "resume": resume,
            "job_description": job_description
        })
        
        # Try to parse as JSON
        import json
        try:
            llm_response = json.loads(llm_response_text)
        except json.JSONDecodeError:
            # If not valid JSON, create a fallback response
            llm_response = {
                "Missing Keywords and Skills": ["Unable to parse - please check resume format"],
                "Profile Summary": llm_response_text[:500] + "..." if len(llm_response_text) > 500 else llm_response_text
            }
    except Exception as e:
        print(f"Warning: LLM analysis failed: {e}")
        llm_response = {
            "Missing Keywords and Skills": ["Analysis temporarily unavailable"],
            "Profile Summary": "Unable to generate profile summary at this time."
        }
    
    final_result = {
        "Job Description Match": round(similarity_percentage, 2),
        "Missing Keywords and Skills": llm_response.get("Missing Keywords and Skills", []),
        "Confidence Score": confidence_score,
        "Profile Summary": llm_response.get("Profile Summary", ""),
        "Should I hire them or not?": decision
    }
    return final_result