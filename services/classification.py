from llm.langchain_setup import evaluate_resume


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
