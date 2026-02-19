from typing import List, Optional, Dict, Any
from uuid import uuid4
import os
import re
import json

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config import GOOGLE_API_KEY
import google.generativeai as genai
from llm.tools import confidence_score_calculation_tool, hiring_decision_tool

genai.configure(api_key=GOOGLE_API_KEY)

# LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.15)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# -------------------------
# Pydantic schemas for robust JSON output
# -------------------------
class JDParsing(BaseModel):
    hard_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    tools_and_technologies: List[str] = Field(default_factory=list)
    experience_requirements: Optional[str] = ""
    education_requirements: Optional[str] = ""
    industry_keywords: List[str] = Field(default_factory=list)

class WorkItem(BaseModel):
    title: Optional[str] = ""
    company: Optional[str] = ""
    duration: Optional[str] = ""
    key_achievements: List[str] = Field(default_factory=list)

class EduItem(BaseModel):
    degree: Optional[str] = ""
    institution: Optional[str] = ""
    field: Optional[str] = ""
    graduation_year: Optional[str] = ""

class ContactInfo(BaseModel):
    name: Optional[str] = ""
    email: Optional[str] = ""
    phone: Optional[str] = ""

class ResumeParsing(BaseModel):
    contact_info: ContactInfo = ContactInfo()
    work_experience: List[WorkItem] = Field(default_factory=list)
    education: List[EduItem] = Field(default_factory=list)
    hard_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    tools_and_technologies: List[str] = Field(default_factory=list)

class KeywordAnalysis(BaseModel):
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    keyword_placement_score: Optional[str] = "0%"
    context_relevance: Optional[str] = "medium"

class ExperienceMatch(BaseModel):
    years_experience: Optional[str] = "0 years"
    experience_level_match: Optional[str] = "0%"
    relevant_experience: Optional[str] = "0%"

class EducationMatch(BaseModel):
    degree_match: Optional[str] = "0%"
    field_relevance: Optional[str] = "0%"

class ATSReadability(BaseModel):
    format_score: Optional[str] = "85%"
    parsing_friendly: Optional[bool] = True
    recommendations: List[str] = Field(default_factory=list)

class OverallAssessment(BaseModel):
    profile_summary: Optional[str] = ""
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    fit_assessment: Optional[str] = "fair"

class AnalysisSchema(BaseModel):
    job_description_parsing: JDParsing
    resume_parsing: ResumeParsing
    keyword_analysis: KeywordAnalysis
    experience_match: ExperienceMatch
    education_match: EducationMatch
    ats_readability: ATSReadability
    overall_assessment: OverallAssessment

parser = PydanticOutputParser(pydantic_object=AnalysisSchema)

classification_prompt = PromptTemplate.from_template("""
You are an advanced ATS (Applicant Tracking System) that follows industry-standard resume parsing and evaluation protocols.

Your task is to perform comprehensive parsing and analysis of both the resume and job description to extract structured data and provide detailed evaluation.

Resume:
{resume}

Job Description:
{job_description}

Instructions:
- Extract structured JSON according to the provided schema.
- Be concise, ensure arrays contain normalized lowercase strings where appropriate.
- Do not include any text outside of JSON.

Format instructions:
{format_instructions}
""").partial(format_instructions=parser.get_format_instructions())

classifier_chain = classification_prompt | llm | parser

# -------------------------
# Vector store setup
# -------------------------
def create_ats_optimized_vector_store():
    """
    Create an ATS-optimized vector store with industry-standard job descriptions
    and resume patterns for better matching according to ATS principles.
    """
    ats_documents = []

    job_description_templates = [
        """Software Engineer - Python, Java, JavaScript, React, Node.js, SQL, MongoDB, AWS, Docker, Kubernetes, 
        Git, Agile, REST APIs, microservices, unit testing, CI/CD, problem-solving, teamwork, communication""",
        """Senior Full Stack Developer - React, Angular, Vue.js, Python, Django, Flask, PostgreSQL, Redis, 
        AWS/Azure, DevOps, leadership, mentoring, system design, scalability, performance optimization""",
        """Frontend Developer - JavaScript, TypeScript, React, Vue, HTML5, CSS3, SASS, Webpack, responsive design,
        cross-browser compatibility, accessibility, Git, Jira, Figma, communication, attention to detail""",
        """Data Scientist - Python, R, SQL, machine learning, deep learning, TensorFlow, PyTorch, scikit-learn,
        pandas, numpy, matplotlib, Jupyter, statistics, data visualization, analytical thinking, problem-solving""",
        """Machine Learning Engineer - Python, TensorFlow, PyTorch, Keras, MLOps, Docker, Kubernetes, AWS/GCP,
        model deployment, data pipelines, feature engineering, A/B testing, monitoring, collaboration""",
        """AI Research Scientist - PhD, machine learning, deep learning, computer vision, NLP, research publications,
        TensorFlow, PyTorch, Python, mathematical modeling, innovation, critical thinking""",
        """DevOps Engineer - AWS, Azure, Docker, Kubernetes, Jenkins, CI/CD, Terraform, Ansible, Linux, BASH,
        monitoring, infrastructure as code, automation, troubleshooting, collaboration""",
        """Cloud Solutions Architect - AWS Certified, Azure, cloud migration, system design, scalability,
        security, cost optimization, leadership, stakeholder management, technical documentation""",
        """Product Manager - MBA preferred, product strategy, roadmap planning, market research, analytics,
        Jira, Confluence, stakeholder management, leadership, communication, data-driven decision making""",
        """Project Manager - PMP certification, Agile, Scrum Master, risk management, budget planning,
        team coordination, stakeholder communication, timeline management, problem-solving""",
        """UI/UX Designer - Figma, Sketch, Adobe Creative Suite, prototyping, user research, wireframing,
        design systems, usability testing, creativity, attention to detail, communication""",
        """QA Engineer - manual testing, automation testing, Selenium, TestNG, bug tracking, test planning,
        attention to detail, analytical thinking, communication, continuous improvement""",
        """Data Engineer - Python, SQL, Apache Spark, Hadoop, ETL, data warehousing, big data, cloud platforms,
        data pipelines, Apache Airflow, problem-solving, analytical thinking""",
        """Security Engineer - CISSP, cybersecurity, network security, penetration testing, risk assessment,
        incident response, security protocols, analytical thinking, attention to detail""",
        """Technical Skills: programming languages, frameworks, databases, cloud platforms, tools, methodologies,
        certifications, years of experience, education background, soft skills, leadership experience""",
        """Professional Experience: full-time employment, project management, team collaboration, client interaction,
        problem-solving, innovation, process improvement, training, mentoring, technical documentation"""
    ]

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

    try:
        vector_store = FAISS.from_documents(ats_documents, embedding_model)
        vector_store.save_local("vector_store")
        print(f"Created ATS-optimized vector store with {len(ats_documents)} documents")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
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

# -------------------------
# Helper functions
# -------------------------
def _extract_percent(value: Any, default: float) -> float:
    """Extract a numeric percent from '85%' or numbers; fallback to default."""
    try:
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"(\d+(\.\d+)?)", str(value))
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return float(default)

def _distance_to_similarity_percent(distance: float) -> float:
    """
    Convert FAISS distance (lower is better) to a bounded similarity percentage [0, 100].
    Uses 1/(1+distance) mapping to avoid negatives and keep scale stable.
    """
    try:
        d = float(distance)
        sim01 = 1.0 / (1.0 + max(0.0, d))
        return max(0.0, min(100.0, sim01 * 100.0))
    except Exception:
        return 50.0

def _index_entities_from_analysis(analysis: AnalysisSchema, context: Dict[str, Any]):
    """
    Take structured analysis and upsert entity documents (skills, certifications, tools, etc.)
    into the vector store with metadata for later retrieval.
    """
    global vector_store

    docs: List[Document] = []

    # Resume entities
    rp = analysis.resume_parsing
    for skill in rp.hard_skills:
        docs.append(Document(page_content=f"skill: {skill}", metadata={**context, "entity_type": "skill", "source": "resume"}))
    for sskill in rp.soft_skills:
        docs.append(Document(page_content=f"soft_skill: {sskill}", metadata={**context, "entity_type": "soft_skill", "source": "resume"}))
    for cert in rp.certifications:
        docs.append(Document(page_content=f"certification: {cert}", metadata={**context, "entity_type": "certification", "source": "resume"}))
    for tool in rp.tools_and_technologies:
        docs.append(Document(page_content=f"tool: {tool}", metadata={**context, "entity_type": "tool", "source": "resume"}))

    # Job description entities
    jp = analysis.job_description_parsing
    for skill in jp.hard_skills:
        docs.append(Document(page_content=f"skill: {skill}", metadata={**context, "entity_type": "skill", "source": "job_description"}))
    for sskill in jp.soft_skills:
        docs.append(Document(page_content=f"soft_skill: {sskill}", metadata={**context, "entity_type": "soft_skill", "source": "job_description"}))
    for cert in jp.certifications:
        docs.append(Document(page_content=f"certification: {cert}", metadata={**context, "entity_type": "certification", "source": "job_description"}))
    for tool in jp.tools_and_technologies:
        docs.append(Document(page_content=f"tool: {tool}", metadata={**context, "entity_type": "tool", "source": "job_description"}))

    if docs:
        vector_store.add_documents(docs)
        vector_store.save_local(vector_store_path)
        print(f"Indexed {len(docs)} structured entities into vector store.")

def _extract_entities_from_single_text(text: str, source_type: str) -> Dict[str, List[str]]:
    """
    Lightweight entity extractor for single uploads (resume or job description).
    Returns dict of lists: hard_skills, soft_skills, certifications, tools_and_technologies.
    """
    mini_prompt = PromptTemplate.from_template("""
Extract the following arrays from the {source_type} text. Respond with strict JSON only:
- hard_skills
- soft_skills
- certifications
- tools_and_technologies

Text:
{text}
""")
    chain = mini_prompt | llm | StrOutputParser()
    raw = chain.invoke({"text": text, "source_type": source_type})
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return {
        "hard_skills": [],
        "soft_skills": [],
        "certifications": [],
        "tools_and_technologies": []
    }

def _index_single_text_entities(text: str, meta: Dict[str, Any]):
    """Index entities from a single text (resume or JD) into vector store."""
    global vector_store
    source_type = meta.get("type", "unknown")
    entities = _extract_entities_from_single_text(text, source_type)
    docs: List[Document] = []
    for skill in entities.get("hard_skills", []):
        docs.append(Document(page_content=f"skill: {skill}", metadata={**meta, "entity_type": "skill"}))
    for sskill in entities.get("soft_skills", []):
        docs.append(Document(page_content=f"soft_skill: {sskill}", metadata={**meta, "entity_type": "soft_skill"}))
    for cert in entities.get("certifications", []):
        docs.append(Document(page_content=f"certification: {cert}", metadata={**meta, "entity_type": "certification"}))
    for tool in entities.get("tools_and_technologies", []):
        docs.append(Document(page_content=f"tool: {tool}", metadata={**meta, "entity_type": "tool"}))

    if docs:
        vector_store.add_documents(docs)
        vector_store.save_local(vector_store_path)
        print(f"Indexed {len(docs)} {source_type} entities into vector store.")

# -------------------------
# Public functions
# -------------------------
def add_resume_to_vector_store(resume_text: str, metadata: dict = None):
    """
    Add a new resume to the existing vector store and auto-extract entities.
    """
    global vector_store

    if metadata is None:
        metadata = {"type": "resume", "added_date": "dynamic"}
    if "resume_id" not in metadata:
        metadata["resume_id"] = str(uuid4())

    try:
        resume_doc = Document(page_content=resume_text, metadata=metadata)
        vector_store.add_documents([resume_doc])
        vector_store.save_local(vector_store_path)
        print("Resume added to vector store successfully")
        _index_single_text_entities(resume_text, metadata)
    except Exception as e:
        print(f"Error adding resume to vector store: {e}")

def add_job_description_to_vector_store(job_desc_text: str, metadata: dict = None):
    """
    Add a new job description to the existing vector store and auto-extract entities.
    """
    global vector_store

    if metadata is None:
        metadata = {"type": "job_description", "added_date": "dynamic"}
    if "job_id" not in metadata:
        metadata["job_id"] = str(uuid4())

    try:
        job_doc = Document(page_content=job_desc_text, metadata=metadata)
        vector_store.add_documents([job_doc])
        vector_store.save_local(vector_store_path)
        print("Job description added to vector store successfully")
        _index_single_text_entities(job_desc_text, metadata)
    except Exception as e:
        print(f"Error adding job description to vector store: {e}")

# -------------------------
# Main evaluation
# -------------------------
def evaluate_resume(resume: str, job_description: str, vector_store_param=None) -> dict:
    """
    Enhanced ATS-style resume evaluation with comprehensive parsing and weighted scoring.
    """
    vs = vector_store_param if vector_store_param is not None else vector_store

    # 1. Calculate FAISS similarity (bounded conversion)
    try:
        faiss_similarity = vs.similarity_search_with_score(job_description, k=1)
        _, distance = faiss_similarity[0]
        base_similarity = _distance_to_similarity_percent(distance)
    except Exception as e:
        print(f"Warning: FAISS similarity calculation failed: {e}")
        base_similarity = 50.0

    # 2. ATS keyword analysis
    from llm.tools import ats_keyword_analyzer
    try:
        keyword_analysis = ats_keyword_analyzer.invoke({
            "resume_text": resume,
            "job_description": job_description
        })
        keyword_match_score = float(keyword_analysis.get('keyword_match_score', base_similarity))
        missing_keywords = keyword_analysis.get('missing_keywords', [])
    except Exception as e:
        print(f"Warning: Keyword analysis failed: {e}")
        keyword_match_score = float(base_similarity)
        missing_keywords = ["Analysis temporarily unavailable"]

    # 3. LLM structured response (strict JSON via Pydantic parser)
    try:
        llm_response: AnalysisSchema = classifier_chain.invoke({
            "resume": resume,
            "job_description": job_description
        })
    except Exception as e:
        print(f"Warning: LLM comprehensive analysis failed: {e}")
        llm_response = AnalysisSchema(
            job_description_parsing=JDParsing(),
            resume_parsing=ResumeParsing(),
            keyword_analysis=KeywordAnalysis(missing_keywords=missing_keywords, keyword_placement_score=f"{keyword_match_score:.1f}%"),
            experience_match=ExperienceMatch(relevant_experience="70%", experience_level_match="70%"),
            education_match=EducationMatch(degree_match="70%", field_relevance="70%"),
            ats_readability=ATSReadability(format_score="85%", parsing_friendly=True, recommendations=[]),
            overall_assessment=OverallAssessment(profile_summary="Unable to generate comprehensive analysis at this time.",
                                                strengths=["Analysis unavailable"],
                                                areas_for_improvement=["Analysis unavailable"],
                                                fit_assessment="unknown")
        )

    # 4. Extract numeric scores for weighted calculation
    try:
        exp_match_val = llm_response.experience_match.experience_level_match or llm_response.experience_match.relevant_experience or "70%"
        experience_match = _extract_percent(exp_match_val, default=base_similarity * 0.9)
        ats_readability = _extract_percent(llm_response.ats_readability.format_score, default=85.0)
    except Exception:
        experience_match = base_similarity * 0.9
        ats_readability = 85.0

    # 5. Calculate confidence score
    confidence_score = confidence_score_calculation_tool.invoke({
        "similarity": base_similarity,
        "keyword_match_score": keyword_match_score,
        "experience_match": experience_match,
        "ats_readability": ats_readability
    })

    # 6. Hiring decision
    decision = hiring_decision_tool.invoke({"confidence": confidence_score})

    # 7. Index structured entities extracted from this run
    context_meta = {
        "analysis_id": str(uuid4()),
        "resume_id": str(uuid4()),
        "job_id": str(uuid4())
    }
    try:
        _index_entities_from_analysis(llm_response, context_meta)
    except Exception as e:
        print(f"Warning: Failed to index structured entities: {e}")

# 8. Final response (UI-friendly and detailed)
    final_result = {
        "Match": f"{keyword_match_score:.1f}%",  # UI alias
        "Job Description Match": f"{base_similarity:.1f}%",
        "Keyword Match Score": f"{keyword_match_score:.1f}%",
        "Experience Match": f"{experience_match:.1f}%",
        "ATS Readability": f"{ats_readability:.1f}%",
        "Confidence Score": float(confidence_score),

        # Provide decision in multiple key styles for template compatibility
        "Hiring Decision": decision,           # human-friendly label
        "hiring_decision": decision,           # snake_case alias
        "decision": decision,                  # short alias

        "Missing Keywords and Skills": llm_response.keyword_analysis.missing_keywords or missing_keywords,
        "Profile Summary": llm_response.overall_assessment.profile_summary or "Analysis completed",
        "Strengths": llm_response.overall_assessment.strengths,
        "Areas for Improvement": llm_response.overall_assessment.areas_for_improvement,
        "Fit Assessment": llm_response.overall_assessment.fit_assessment,
        "Job Description Parsing": llm_response.job_description_parsing.model_dump(),
        "Resume Parsing": llm_response.resume_parsing.model_dump(),
        "Detailed Analysis": llm_response.keyword_analysis.model_dump(),
        "Score Breakdown": {
            "Base Similarity": f"{base_similarity:.1f}%",
            "Keyword Matching (60% weight)": f"{keyword_match_score:.1f}%",
            "Experience Match (25% weight)": f"{experience_match:.1f}%",
            "ATS Readability (15% weight)": f"{ats_readability:.1f}%",
            "Final Weighted Score": f"{float(confidence_score):.1f}%"
        }
    }

    return final_result


async def evaluate_resume_async(resume: str, job_description: str, vector_store_param=None) -> dict:
    """
    Async version: Enhanced ATS-style resume evaluation with comprehensive parsing and weighted scoring.
    """
    import asyncio
    
    # Run the synchronous evaluation in an executor to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: evaluate_resume(resume, job_description, vector_store_param)
    )
    return result