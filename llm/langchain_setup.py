from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from config import GOOGLE_API_KEY
import google.generativeai as genai
from tools import confidence_score_calculation_tool, hiring_decision_tool
from langchain.vectorstores import FAISS
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.15)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")



vector_store = FAISS.load_local("vector_store", embedding_model)

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

def evaluate_resume(resume:str, job_description:str,vector_store)-> dict:
    """
    Evaluate a resume against a job description using FAISS similarity for deterministic scoring
    and an LLM for qualitative analysis.

    Args:
        resume (str): The candidate's resume.
        job_description (str): The job description for the position.
        vector_store: The vector store for similarity calculations.

    Returns:
        dict: Evaluation results including job description match, missing skills, confidence score, profile summary, and hiring decision.
    """
    # Calculate FAISS similarity
    faiss_similarity = vector_store.similarity_search_with_score(job_description, k=1)
    _, distance = faiss_similarity[0]
    similarity_percentage = 100 - (distance * 100)

    # confidence score calculation
    confidence_score = confidence_score_calculation_tool.invoke(similarity_percentage)

    #hiring decision
    decision= hiring_decision_tool.invoke(confidence_score)
    llm_response = classifier_chain.invoke({
        "resume": resume,
        "job_description": job_description
    })
    final_result = {
        "Job Description Match": similarity_percentage,
        "Missing Keywords and Skills": llm_response.get("Missing Keywords and Skills", []),
        "Confidence Score": confidence_score,
        "Profile Summary": llm_response.get("Profile Summary", ""),
        "Should I hire them or not?": decision
    }
    return final_result