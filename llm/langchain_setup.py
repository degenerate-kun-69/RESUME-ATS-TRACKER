from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from config import GOOGLE_API_KEY
import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.15)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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
1. Identify if the candidate is a good match for the position
2. Calculate a job description match percentage
3. Estimate a confidence score (0-100)
4. Highlight any missing keywords and skills
5. Provide a concise profile summary
6. Decide: Should they be hired?

Respond strictly in this JSON format:
{{
  "Job Description Match": "percentage%",
  "Missing Keywords and Skills": ["skill1", "skill2", "..."],
  "Confidence Score": 0-100,
  "Profile Summary": "detailed summary...",
  "Should I hire them or not?": "yes/no"
}}
""")

classifier_chain = classification_prompt | llm | StrOutputParser()
