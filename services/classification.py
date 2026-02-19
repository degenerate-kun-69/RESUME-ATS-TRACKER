from llm.langchain_setup import evaluate_resume, evaluate_resume_async
from utils.redis_cache import cache_result


def classify_resume(resume_text, job_description):
    """
    Classify and evaluate a resume against a job description.
    
    Args:
        resume_text (str): The extracted text from the resume
        job_description (str): The job description text
        
    Returns:
        dict: Classification results with match percentage, confidence score, etc.
    """
    return evaluate_resume(resume_text, job_description)


@cache_result(prefix="resume_analysis", ttl=3600)  # Cache for 1 hour
async def classify_resume_async(resume_text: str, job_description: str):
    """
    Async version: Classify and evaluate a resume against a job description with caching.
    
    Args:
        resume_text (str): The extracted text from the resume
        job_description (str): The job description text
        
    Returns:
        dict: Classification results with match percentage, confidence score, etc.
    """
    return await evaluate_resume_async(resume_text, job_description)
