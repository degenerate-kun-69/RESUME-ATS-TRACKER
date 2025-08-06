from dotenv import load_dotenv
load_dotenv()

import os
import json
from flask import Flask, request, render_template, redirect, url_for, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import pymupdf

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './temp'

# LangChain setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.15
)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Enhanced Classification Prompt (merged from both files)
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

# Create the classification chain
classifier_chain = (
    classification_prompt
    | llm
    | StrOutputParser()
)

# Helper functions from both files
def get_gemini_response_direct(input_text, pdf_content, prompt):
    """Direct Gemini API call (from app.py)"""
    model = genai.GenerativeModel(model_name='gemini-2.5-pro')
    generation_config = {
        "temperature": 0.15,
        "top_p": 0.95,
        "max_output_tokens": 1024,
        "stop_sequences": ["\n"],
        "top_k": 40
    }
    response = model.generate_content(
        [input_text, pdf_content, prompt],
        generation_config=generation_config
    )
    return response.text

def input_pdf_to_text(pdf_file):
    """Extract text from PDF using PyMuPDF (from app.py)"""
    pdf_document = pymupdf.open(pdf_file)
    pdf_text = ""
    for page in pdf_document:
        pdf_text += page.get_text()
    return pdf_text

def extract_text_from_pdf(file_path):
    """Extract text from PDF using LangChain (from app2.py)"""
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

def build_job_vector_index(job_descriptions):
    """Build FAISS vector index for job recommendations"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.create_documents(job_descriptions)
    return FAISS.from_documents(docs, embedding=embedding_model)

def get_job_recommendations(resume_text, faiss_index, top_k=3):
    """Get job recommendations based on resume using FAISS similarity search"""
    return faiss_index.similarity_search(resume_text, k=top_k)

def generate_job_recommendations(job_description, resume_text, top_k=5):
    """Generate job recommendations based on the job description and resume"""
    # Create variations of the job description for recommendations
    job_variations = [
        f"Senior {job_description}",
        f"Junior {job_description}",
        f"Entry Level {job_description}",
        f"Lead {job_description}",
        f"Remote {job_description}",
    ]
    
    # Extract key skills and technologies from job description
    job_lower = job_description.lower()
    tech_keywords = []
    common_techs = [
        'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js', 'express',
        'django', 'flask', 'spring', 'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'linux',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi', 'excel',
        'agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'api', 'rest', 'graphql'
    ]
    
    for tech in common_techs:
        if tech in job_lower:
            tech_keywords.append(tech)
    
    # Generate related job recommendations based on extracted technologies
    if tech_keywords:
        related_jobs = []
        if any(tech in tech_keywords for tech in ['python', 'machine learning', 'tensorflow', 'pytorch']):
            related_jobs.extend([
                "Data Scientist: Python, ML, pandas, scikit-learn, statistics, data visualization",
                "ML Engineer: TensorFlow, PyTorch, MLOps, model deployment, cloud platforms",
                "AI Research Scientist: Deep learning, research, publications, algorithm development"
            ])
        
        if any(tech in tech_keywords for tech in ['javascript', 'react', 'angular', 'vue']):
            related_jobs.extend([
                "Frontend Developer: React, JavaScript, TypeScript, CSS, responsive design",
                "Full Stack Developer: React, Node.js, JavaScript, database integration",
                "UI/UX Developer: Frontend frameworks, design systems, user experience"
            ])
        
        if any(tech in tech_keywords for tech in ['java', 'spring', 'microservices']):
            related_jobs.extend([
                "Backend Developer: Java, Spring Boot, microservices, REST APIs",
                "Software Engineer: Java, system design, scalable applications",
                "Enterprise Developer: Java EE, enterprise applications, integration"
            ])
        
        if any(tech in tech_keywords for tech in ['aws', 'azure', 'docker', 'kubernetes']):
            related_jobs.extend([
                "DevOps Engineer: AWS, Docker, Kubernetes, CI/CD, infrastructure automation",
                "Cloud Engineer: Cloud platforms, infrastructure, serverless, monitoring",
                "Site Reliability Engineer: System reliability, monitoring, automation"
            ])
        
        if any(tech in tech_keywords for tech in ['sql', 'mysql', 'postgresql', 'mongodb']):
            related_jobs.extend([
                "Database Administrator: SQL, database optimization, backup strategies",
                "Data Engineer: ETL pipelines, data warehousing, big data technologies",
                "Backend Developer: Database design, API development, server-side logic"
            ])
        
        # If we have related jobs, use them; otherwise fall back to general recommendations
        if related_jobs:
            job_variations.extend(related_jobs[:3])  # Add top 3 related jobs
    
    # If no specific technologies found, add general job categories
    if not tech_keywords:
        job_variations.extend([
            "Software Developer: Programming, problem-solving, software development lifecycle",
            "Project Manager: Project planning, team coordination, stakeholder management",
            "Business Analyst: Requirements analysis, process improvement, documentation",
            "Quality Assurance Engineer: Testing, automation, quality processes",
            "Product Manager: Product strategy, market research, feature planning"
        ])
    
    # Build vector index and get recommendations
    try:
        job_index = build_job_vector_index(job_variations)
        recommended = get_job_recommendations(resume_text, job_index, top_k)
        return [doc.page_content for doc in recommended]
    except Exception as e:
        # Fallback to simple text-based recommendations
        return job_variations[:top_k]

def parse_json_response(response_text):
    """Parse JSON response with error handling"""
    try:
        # Try to extract JSON from the response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        elif response_text.strip().startswith("{"):
            json_text = response_text.strip()
        else:
            # If no clear JSON format, try to find JSON-like content
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
            else:
                raise ValueError("No JSON found in response")
        
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as e:
        # Return a fallback structure if JSON parsing fails
        return {
            "Job Description Match": "Unable to parse",
            "Missing Keywords and Skills": ["Analysis failed"],
            "Confidence Score": 0,
            "Profile Summary": f"Error parsing response: {str(e)}",
            "Should I hire them or not?": "no"
        }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """Main analysis route (enhanced from app2.py)"""
    if 'resume' not in request.files:
        return redirect(url_for('index'))

    resume_file = request.files['resume']
    job_input_type = request.form.get('job_input_type', 'text')

    if resume_file.filename == '':
        return redirect(url_for('index'))

    # Get job description based on input type
    if job_input_type == 'text':
        if 'job_description' not in request.form or not request.form['job_description'].strip():
            return render_template('index.html', error="Please provide a job description")
        job_description = request.form['job_description']
    elif job_input_type == 'file':
        if 'job_description_file' not in request.files:
            return render_template('index.html', error="Please upload a job description file")
        job_desc_file = request.files['job_description_file']
        if job_desc_file.filename == '':
            return render_template('index.html', error="Please select a job description file")
        
        # Save and extract job description from PDF
        job_desc_path = os.path.join(app.config['UPLOAD_FOLDER'], f"job_desc_{job_desc_file.filename}")
        job_desc_file.save(job_desc_path)
        
        try:
            job_description = extract_text_from_pdf(job_desc_path)
            if not job_description.strip():
                return render_template('index.html', error="Could not extract text from job description PDF")
        except Exception as e:
            return render_template('index.html', error=f"Error reading job description PDF: {str(e)}")
        finally:
            # Clean up job description file
            if os.path.exists(job_desc_path):
                os.remove(job_desc_path)
    else:
        return render_template('index.html', error="Invalid job input type")

    # Save uploaded resume file
    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_path)

    try:
        # Extract resume content using LangChain method
        resume_text = extract_text_from_pdf(resume_path)

        # Run classification using LangChain
        input_data = {
            "resume": resume_text,
            "job_description": job_description
        }
        
        classification_response = classifier_chain.invoke(input_data)
        classification_result = parse_json_response(classification_response)

        # Generate job recommendations based on the actual job description
        recommendations = generate_job_recommendations(job_description, resume_text)

        # Clean up temporary file
        os.remove(resume_path)

        return render_template('index.html',
                             result=classification_result,
                             recommendations=recommendations)
                             
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(resume_path):
            os.remove(resume_path)
        return render_template('index.html', error=f"Analysis failed: {str(e)}")

@app.route('/api/analyze', methods=['POST'])
def api_analyze_resume():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        if not data or 'resume_text' not in data or 'job_description' not in data:
            return jsonify({"error": "Missing resume_text or job_description"}), 400

        resume_text = data['resume_text']
        job_description = data['job_description']

        # Run classification
        input_data = {
            "resume": resume_text,
            "job_description": job_description
        }
        
        classification_response = classifier_chain.invoke(input_data)
        classification_result = parse_json_response(classification_response)

        return jsonify({
            "success": True,
            "analysis": classification_result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Resume ATS Tracker"})

if __name__ == '__main__':
    # Create temp directory if it doesn't exist
    os.makedirs('./temp', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)