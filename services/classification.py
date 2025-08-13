from llm.langchain_setup import evaluate_resume


def classify_resume(resume_text, job_description):
    input_data = {
        "resume": resume_text,
        "job_description": job_description
    }
    return evaluate_resume(**input_data)
