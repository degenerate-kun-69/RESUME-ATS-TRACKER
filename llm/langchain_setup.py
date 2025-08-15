from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from config import GOOGLE_API_KEY
import google.generativeai as genai
from llm.tools import confidence_score_calculation_tool, hiring_decision_tool
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.15)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_ats_optimized_vector_store():
    """
    Create an ATS-optimized vector store with industry-standard job descriptions
    and resume patterns for better matching according to ATS principles.
    """
    from langchain.schema import Document
    
    # ATS-optimized document collection
    ats_documents = []
    
    # 1. Industry-standard job description templates
    job_description_templates = [
        # Software Development
        """Software Engineer - Python, Java, JavaScript, React, Node.js, SQL, MongoDB, AWS, Docker, Kubernetes, 
        Git, Agile, REST APIs, microservices, unit testing, CI/CD, problem-solving, teamwork, communication""",
        
        """Senior Full Stack Developer - React, Angular, Vue.js, Python, Django, Flask, PostgreSQL, Redis, 
        AWS/Azure, DevOps, leadership, mentoring, system design, scalability, performance optimization""",
        
        """Frontend Developer - JavaScript, TypeScript, React, Vue, HTML5, CSS3, SASS, Webpack, responsive design,
        cross-browser compatibility, accessibility, Git, Jira, Figma, communication, attention to detail""",
        
        # Data Science & AI
        """Data Scientist - Python, R, SQL, machine learning, deep learning, TensorFlow, PyTorch, scikit-learn,
        pandas, numpy, matplotlib, Jupyter, statistics, data visualization, analytical thinking, problem-solving""",
        
        """Machine Learning Engineer - Python, TensorFlow, PyTorch, Keras, MLOps, Docker, Kubernetes, AWS/GCP,
        model deployment, data pipelines, feature engineering, A/B testing, monitoring, collaboration""",
        
        """AI Research Scientist - PhD, machine learning, deep learning, computer vision, NLP, research publications,
        TensorFlow, PyTorch, Python, mathematical modeling, innovation, critical thinking""",
        
        # DevOps & Cloud
        """DevOps Engineer - AWS, Azure, Docker, Kubernetes, Jenkins, CI/CD, Terraform, Ansible, Linux, BASH,
        monitoring, infrastructure as code, automation, troubleshooting, collaboration""",
        
        """Cloud Solutions Architect - AWS Certified, Azure, cloud migration, system design, scalability,
        security, cost optimization, leadership, stakeholder management, technical documentation""",
        
        # Business & Management
        """Product Manager - MBA preferred, product strategy, roadmap planning, market research, analytics,
        Jira, Confluence, stakeholder management, leadership, communication, data-driven decision making""",
        
        """Project Manager - PMP certification, Agile, Scrum Master, risk management, budget planning,
        team coordination, stakeholder communication, timeline management, problem-solving""",
        
        # Design & UX
        """UI/UX Designer - Figma, Sketch, Adobe Creative Suite, prototyping, user research, wireframing,
        design systems, usability testing, creativity, attention to detail, communication""",
        
        # Quality Assurance
        """QA Engineer - manual testing, automation testing, Selenium, TestNG, bug tracking, test planning,
        attention to detail, analytical thinking, communication, continuous improvement""",
        
        # Data Engineering
        """Data Engineer - Python, SQL, Apache Spark, Hadoop, ETL, data warehousing, big data, cloud platforms,
        data pipelines, Apache Airflow, problem-solving, analytical thinking""",
        
        # Cybersecurity
        """Security Engineer - CISSP, cybersecurity, network security, penetration testing, risk assessment,
        incident response, security protocols, analytical thinking, attention to detail""",
        
        # Generic skill combinations for broader matching
        """Technical Skills: programming languages, frameworks, databases, cloud platforms, tools, methodologies,
        certifications, years of experience, education background, soft skills, leadership experience""",
        
        """Professional Experience: full-time employment, project management, team collaboration, client interaction,
        problem-solving, innovation, process improvement, training, mentoring, technical documentation"""
    ]
    
    # 2. Common resume patterns and skill combinations
    resume_patterns = [
        """Software Developer with 3+ years experience in Python, JavaScript, React, Node.js, MySQL, Git.
        Bachelor's degree in Computer Science. Strong problem-solving and communication skills.""",
        
        """Data Scientist with PhD in Statistics, expertise in machine learning, Python, R, TensorFlow,
        data analysis, statistical modeling, research publications, presentation skills.""",
        
        """DevOps Engineer with AWS certification, Docker, Kubernetes, Jenkins, CI/CD, Linux administration,
        infrastructure automation, monitoring, troubleshooting, team collaboration.""",
        
        """Full Stack Developer proficient in React, Python, Django, PostgreSQL, REST APIs, Agile methodology,
        version control, testing, debugging, continuous learning mindset.""",
        
        """Product Manager with MBA, 5+ years experience in product strategy, market analysis, roadmap planning,
        stakeholder management, cross-functional team leadership, data-driven decisions."""
    ]
    
    # 3. Add all documents to collection
    for i, job_template in enumerate(job_description_templates):
        ats_documents.append(Document(
            page_content=job_template,
            metadata={"type": "job_description", "category": f"job_template_{i}"}
        ))
    
    for i, resume_pattern in enumerate(resume_patterns):
        ats_documents.append(Document(
            page_content=resume_pattern,
            metadata={"type": "resume_pattern", "category": f"resume_pattern_{i}"}
        ))
    
    # 4. Create and save vector store
    try:
        vector_store = FAISS.from_documents(ats_documents, embedding_model)
        vector_store.save_local("vector_store")
        print(f"Created ATS-optimized vector store with {len(ats_documents)} documents")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        # Fallback to simple dummy document
        dummy_doc = Document(page_content="Software Engineer Python JavaScript", metadata={"type": "fallback"})
        vector_store = FAISS.from_documents([dummy_doc], embedding_model)
        vector_store.save_local("vector_store")
        return vector_store

# Initialize vector store - create empty one if it doesn't exist
vector_store_path = "vector_store"
if os.path.exists(vector_store_path):
    try:
        vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Warning: Could not load existing vector store: {e}")
        print("Creating ATS-optimized vector store...")
        vector_store = create_ats_optimized_vector_store()
else:
    print("Vector store not found. Creating ATS-optimized vector store...")
    vector_store = create_ats_optimized_vector_store()


def add_resume_to_vector_store(resume_text: str, metadata: dict = None):
    """
    Add a new resume to the existing vector store for future matching.
    
    Args:
        resume_text (str): The resume text to add
        metadata (dict): Optional metadata for the resume
    """
    global vector_store
    
    if metadata is None:
        metadata = {"type": "resume", "added_date": "dynamic"}
    
    try:
        resume_doc = Document(page_content=resume_text, metadata=metadata)
        vector_store.add_documents([resume_doc])
        vector_store.save_local(vector_store_path)
        print("Resume added to vector store successfully")
    except Exception as e:
        print(f"Error adding resume to vector store: {e}")


def add_job_description_to_vector_store(job_desc_text: str, metadata: dict = None):
    """
    Add a new job description to the existing vector store for future matching.
    
    Args:
        job_desc_text (str): The job description text to add
        metadata (dict): Optional metadata for the job description
    """
    global vector_store
    
    if metadata is None:
        metadata = {"type": "job_description", "added_date": "dynamic"}
    
    try:
        job_doc = Document(page_content=job_desc_text, metadata=metadata)
        vector_store.add_documents([job_doc])
        vector_store.save_local(vector_store_path)
        print("Job description added to vector store successfully")
    except Exception as e:
        print(f"Error adding job description to vector store: {e}")

classification_prompt = PromptTemplate.from_template("""
You are an advanced ATS (Applicant Tracking System) that follows industry-standard resume parsing and evaluation protocols.

Your task is to perform comprehensive parsing and analysis of both the resume and job description to extract structured data and provide detailed evaluation.

Resume:
{resume}

Job Description:
{job_description}

PARSING REQUIREMENTS:

1. JOB DESCRIPTION PARSING - Extract and categorize:
   - Hard skills (e.g., Python, Azure, TensorFlow, Java, React)
   - Soft skills (e.g., leadership, communication, teamwork)
   - Certifications (e.g., AWS Certified Solutions Architect, PMP)
   - Tools & technologies (e.g., Jira, Figma, Docker, Kubernetes)
   - Experience requirements (years, level)
   - Education requirements
   - Industry-specific keywords

2. RESUME PARSING - Break down into structured data:
   - Contact information
   - Work experience (titles, companies, duration, achievements)
   - Education (degrees, institutions, fields of study)
   - Skills categorized by type
   - Certifications with expiration dates
   - Projects and achievements
   - Keywords placement and context

3. ANALYSIS REQUIREMENTS:
   - Keyword matching with context weighting
   - Section-based scoring (Skills section = full points, buried mentions = less weight)
   - Experience match evaluation
   - Education alignment assessment
   - Certification relevance
   - ATS-readability factors

Respond strictly in this JSON format:
{{
  "job_description_parsing": {{
    "hard_skills": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"], 
    "certifications": ["cert1", "cert2"],
    "tools_and_technologies": ["tool1", "tool2"],
    "experience_requirements": "X years in relevant field",
    "education_requirements": "Bachelor's degree in relevant field",
    "industry_keywords": ["keyword1", "keyword2"]
  }},
  "resume_parsing": {{
    "contact_info": {{"name": "Name", "email": "email", "phone": "phone"}},
    "work_experience": [
      {{"title": "Job Title", "company": "Company", "duration": "Duration", "key_achievements": ["achievement1"]}}
    ],
    "education": [
      {{"degree": "Degree", "institution": "Institution", "field": "Field", "graduation_year": "Year"}}
    ],
    "hard_skills": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"],
    "certifications": ["cert1", "cert2"],
    "tools_and_technologies": ["tool1", "tool2"]
  }},
  "keyword_analysis": {{
    "matched_keywords": ["keyword1", "keyword2"],
    "missing_keywords": ["missing1", "missing2"],
    "keyword_placement_score": "XX%",
    "context_relevance": "high/medium/low"
  }},
  "experience_match": {{
    "years_experience": "X years",
    "experience_level_match": "XX%",
    "relevant_experience": "XX%"
  }},
  "education_match": {{
    "degree_match": "XX%", 
    "field_relevance": "XX%"
  }},
  "ats_readability": {{
    "format_score": "XX%",
    "parsing_friendly": true/false,
    "recommendations": ["improvement1", "improvement2"]
  }},
  "overall_assessment": {{
    "profile_summary": "Comprehensive candidate profile summary based on structured analysis",
    "strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["area1", "area2"],
    "fit_assessment": "excellent/good/fair/poor"
  }}
}}
""")

classifier_chain = classification_prompt | llm | StrOutputParser()

def evaluate_resume(resume:str, job_description:str, vector_store_param=None)-> dict:
    """
    Enhanced ATS-style resume evaluation with comprehensive parsing and weighted scoring.

    Args:
        resume (str): The candidate's resume.
        job_description (str): The job description for the position.
        vector_store_param: The vector store for similarity calculations.

    Returns:
        dict: Evaluation results including job description match, missing skills, confidence score, profile summary, and hiring decision.
    """
    # Use global vector_store if none provided
    vs = vector_store_param if vector_store_param is not None else vector_store
        
    # 1. Calculate FAISS similarity (base similarity)
    try:
        faiss_similarity = vs.similarity_search_with_score(job_description, k=1)
        _, distance = faiss_similarity[0]
        base_similarity = max(0, min(100, 100 - (distance * 100)))
    except Exception as e:
        print(f"Warning: FAISS similarity calculation failed: {e}")
        base_similarity = 50.0  # Default fallback
    # ATS keyword analysis
    from llm.tools import ats_keyword_analyzer, confidence_score_calculation_tool, hiring_decision_tool
    try:
        keyword_analysis = ats_keyword_analyzer.invoke({
            "resume_text": resume, 
            "job_description": job_description
        })
        keyword_match_score = keyword_analysis.get('keyword_match_score', base_similarity)
        missing_keywords = keyword_analysis.get('missing_keywords', [])
    except Exception as e:
        print(f"Warning: Keyword analysis failed: {e}")
        keyword_match_score = base_similarity
        missing_keywords = ["Analysis temporarily unavailable"]
    # Get LLM response
    try:
        llm_response_text = classifier_chain.invoke({
            "resume": resume,
            "job_description": job_description
        })
        
        # Try to parse as JSON
        import json
        import re
        try:
            # First try direct JSON parsing
            llm_response = json.loads(llm_response_text)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response_text, re.DOTALL)
                if json_match:
                    llm_response = json.loads(json_match.group(1))
                else:
                    # Try to find any JSON-like structure
                    json_match = re.search(r'\{[^{}]*"Missing Keywords and Skills"[^{}]*\}', llm_response_text, re.DOTALL)
                    if json_match:
                        llm_response = json.loads(json_match.group(0))
                    else:
                        raise json.JSONDecodeError("No valid JSON found", llm_response_text, 0)
            except json.JSONDecodeError:
                # If still can't parse, try to extract information manually
                missing_skills = []
                profile_summary = llm_response_text
                
                # Try to extract missing skills list
                skills_match = re.search(r'Missing Keywords and Skills[\'"]?\s*:\s*\[(.*?)\]', llm_response_text, re.DOTALL)
                if skills_match:
                    skills_text = skills_match.group(1)
                    missing_skills = [skill.strip().strip('"\'') for skill in skills_text.split(',') if skill.strip()]
                
                # Try to extract profile summary
                summary_match = re.search(r'Profile Summary[\'"]?\s*:\s*[\'"]([^\'\"]*)[\'"]', llm_response_text, re.DOTALL)
                if summary_match:
                    profile_summary = summary_match.group(1)
                
                llm_response = {
                    "overall_assessment": {
                        "profile_summary": "Analysis completed - detailed parsing temporarily unavailable",
                        "strengths": ["See resume content"],
                        "areas_for_improvement": ["Detailed analysis unavailable"],
                        "fit_assessment": "fair"
                    },
                    "keyword_analysis": {
                        "missing_keywords": missing_keywords,
                        "keyword_placement_score": f"{keyword_match_score:.1f}%"
                    }
                }
    except Exception as e:
        print(f"Warning: LLM comprehensive analysis failed: {e}")
        llm_response = {
            "overall_assessment": {
                "profile_summary": "Unable to generate comprehensive analysis at this time.",
                "strengths": ["Analysis unavailable"],
                "areas_for_improvement": ["Analysis unavailable"],
                "fit_assessment": "unknown"
            }
        }

    # 4. Extract scores for weighted calculation
    try:
        # Extract experience match percentage
        experience_data = llm_response.get('experience_match', {})
        experience_match = float(re.search(r'(\d+)', str(experience_data.get('relevant_experience', '70%'))).group(1)) if re.search(r'(\d+)', str(experience_data.get('relevant_experience', '70%'))) else 70.0
        
        # Extract ATS readability score
        ats_data = llm_response.get('ats_readability', {})
        ats_readability = float(re.search(r'(\d+)', str(ats_data.get('format_score', '85%'))).group(1)) if re.search(r'(\d+)', str(ats_data.get('format_score', '85%'))) else 85.0
        
    except (ValueError, AttributeError):
        # Use defaults if extraction fails
        experience_match = base_similarity * 0.9
        ats_readability = 85.0

    # 5. Calculate comprehensive ATS confidence score
    confidence_score = confidence_score_calculation_tool.invoke({
        "similarity": base_similarity,
        "keyword_match_score": keyword_match_score,
        "experience_match": experience_match,
        "ats_readability": ats_readability
    })

    # 6. Make hiring decision
    decision = hiring_decision_tool.invoke({"confidence": confidence_score})
    
    # 7. Compile comprehensive results
    final_result = {
        # Core ATS Metrics
        "Job Description Match": f"{base_similarity:.1f}%",
        "Keyword Match Score": f"{keyword_match_score:.1f}%",
        "Experience Match": f"{experience_match:.1f}%", 
        "ATS Readability": f"{ats_readability:.1f}%",
        "Confidence Score": confidence_score,
        "Should I hire them or not?": decision,
        
        # Detailed Analysis
        "Missing Keywords and Skills": llm_response.get('keyword_analysis', {}).get('missing_keywords', missing_keywords),
        "Profile Summary": llm_response.get('overall_assessment', {}).get('profile_summary', "Analysis completed"),
        "Strengths": llm_response.get('overall_assessment', {}).get('strengths', []),
        "Areas for Improvement": llm_response.get('overall_assessment', {}).get('areas_for_improvement', []),
        "Fit Assessment": llm_response.get('overall_assessment', {}).get('fit_assessment', "fair"),
        
        # Structured Data (for advanced features)
        "Job Description Parsing": llm_response.get('job_description_parsing', {}),
        "Resume Parsing": llm_response.get('resume_parsing', {}),
        "Detailed Analysis": llm_response.get('keyword_analysis', {}),
        
        # Scoring Breakdown
        "Score Breakdown": {
            "Base Similarity": f"{base_similarity:.1f}%",
            "Keyword Matching (60% weight)": f"{keyword_match_score:.1f}%",
            "Experience Match (25% weight)": f"{experience_match:.1f}%",
            "ATS Readability (15% weight)": f"{ats_readability:.1f}%",
            "Final Weighted Score": f"{confidence_score:.1f}%"
        }
    }
    
    return final_result