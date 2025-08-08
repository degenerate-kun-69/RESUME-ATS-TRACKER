from llm.langchain_setup import classifier_chain

def classify_resume(resume_text, job_description):
    input_data = {
        "resume": resume_text,
        "job_description": job_description
    }
    return classifier_chain.invoke(input_data)
