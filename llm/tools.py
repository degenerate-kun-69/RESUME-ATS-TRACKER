from langchain_core.tools import tool
import numpy as np
import re
from typing import Dict, List, Tuple


@tool("ATS_keyword_analyzer")
def ats_keyword_analyzer(resume_text: str, job_description: str) -> Dict:
    """
    Analyze keywords and their context placement according to ATS standards.
    
    Args:
        resume_text (str): The candidate's resume text
        job_description (str): The job description text
        
    Returns:
        Dict: Analysis results with weighted keyword matching
    """
    # Enhanced keyword categories with more comprehensive lists
    keyword_categories = {
        'hard_skills': [
            # Programming Languages
            'python', 'java', 'javascript', 'c++', 'c', 'sql', 'r',
            # AI/ML Technologies
            'machine learning', 'artificial intelligence', 'ai', 'ml', 'deep learning',
            'langchain', 'autogen', 'crewai', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'hugging face', 'transformers', 'llm', 'large language models', 'nlp',
            'natural language processing', 'prompt engineering', 'rag', 'retrieval augmented generation',
            'lstm', 'cnn', 'computer vision', 'data science',
            # Cloud & Deployment
            'azure', 'aws', 'gcp', 'docker', 'kubernetes', 'serverless', 'cloud deployment',
            'microsoft azure', 'app services', 'azure functions', 'cosmosdb',
            # Databases
            'mongodb', 'oracle', 'postgresql', 'mysql', 'database',
            # Frameworks & Tools
            'git', 'github', 'colab', 'jupyter', 'ollama', 'tavily', 'geoapify',
            # Data & Analytics
            'big data', 'data structures', 'algorithms', 'analytics', 'optimization'
        ],
        'soft_skills': [
            'problem-solving', 'analytical thinking', 'communication', 'teamwork', 
            'collaboration', 'cross-functional', 'leadership', 'research', 'learning',
            'creative', 'adaptable', 'organized', 'detail-oriented', 'innovative',
            'critical thinking', 'decision-making', 'troubleshooting'
        ],
        'certifications': [
            'bachelor', 'b.tech', 'computer science', 'ai/ml', 'mathematics',
            'machine learning specialization', 'stanford', 'coursera', 'deeplearning.ai',
            'java certification', 'mongodb', 'oracle academy', 'udemy'
        ],
        'tools': [
            'git', 'docker', 'colab', 'jupyter', 'ollama', 'lm studio', 'github',
            'azure portal', 'tavily api', 'geoapify api', 'data guard', 'version control'
        ],
        'experience_terms': [
            'intern', 'experience', 'projects', 'developed', 'built', 'integrated',
            'deployed', 'implemented', 'optimized', 'automated', 'upgraded',
            'fine-tuned', 'worked with', 'applied', 'explored', 'achieved'
        ]
    }
    
    print(f"DEBUG: Starting keyword analysis...")
    print(f"DEBUG: Resume length: {len(resume_text)} characters")
    print(f"DEBUG: Job description length: {len(job_description)} characters")
    
    # Extract keywords from job description
    job_keywords = extract_keywords_from_text(job_description.lower(), keyword_categories)
    print(f"DEBUG: Job keywords found: {job_keywords}")
    
    # Analyze resume sections
    resume_analysis = analyze_resume_sections(resume_text.lower())
    print(f"DEBUG: Resume analysis structure: {list(resume_analysis.keys())}")
    
    # Calculate weighted keyword matching
    keyword_match_score = calculate_keyword_match_score(job_keywords, resume_analysis)
    print(f"DEBUG: Calculated keyword match score: {keyword_match_score}")
    
    # Find missing keywords
    missing_keywords = find_missing_keywords(job_keywords, resume_analysis)
    print(f"DEBUG: Missing keywords: {missing_keywords}")
    
    return {
        'job_keywords': job_keywords,
        'resume_analysis': resume_analysis,
        'keyword_match_score': keyword_match_score,
        'missing_keywords': missing_keywords
    }


def extract_keywords_from_text(text: str, categories: Dict) -> Dict:
    """Extract categorized keywords from text with fuzzy matching"""
    found_keywords = {'hard_skills': [], 'soft_skills': [], 'certifications': [], 'tools': [], 'experience_terms': []}
    
    # Convert text to lowercase for matching
    text_lower = text.lower()
    
    for category, keywords in categories.items():
        if category not in found_keywords:
            found_keywords[category] = []
            
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Direct substring match
            if keyword_lower in text_lower:
                found_keywords[category].append(keyword)
                continue
            
            # Handle compound terms and variations
            keyword_parts = keyword_lower.split()
            if len(keyword_parts) > 1:
                # Check if all parts of compound keyword exist
                if all(part in text_lower for part in keyword_parts):
                    found_keywords[category].append(keyword)
                    continue
            
            # Handle common variations and abbreviations
            variations = {
                'machine learning': ['ml', 'machine-learning', 'machinelearning'],
                'artificial intelligence': ['ai', 'artificial-intelligence'],
                'deep learning': ['dl', 'deep-learning', 'deeplearning'],
                'large language models': ['llm', 'llms', 'large language model'],
                'natural language processing': ['nlp', 'natural-language-processing'],
                'computer science': ['cs', 'computer-science', 'comp sci'],
                'problem-solving': ['problem solving', 'problemsolving'],
                'retrieval augmented generation': ['rag'],
                'hugging face': ['huggingface', 'hf'],
                'microsoft azure': ['azure', 'ms azure'],
                'version control': ['git', 'github', 'gitlab']
            }
            
            if keyword_lower in variations:
                for variation in variations[keyword_lower]:
                    if variation in text_lower:
                        found_keywords[category].append(keyword)
                        break
    
    # Remove duplicates while preserving order
    for category in found_keywords:
        found_keywords[category] = list(dict.fromkeys(found_keywords[category]))
    
    print(f"DEBUG: Extracted keywords for text snippet: {found_keywords}")
    return found_keywords


def analyze_resume_sections(resume_text: str) -> Dict:
    """Analyze resume sections and keyword placement"""
    sections = {
        'skills': {'keywords': {'hard_skills': [], 'soft_skills': [], 'certifications': [], 'tools': [], 'experience_terms': []}, 'weight': 1.0},
        'experience': {'keywords': {'hard_skills': [], 'soft_skills': [], 'certifications': [], 'tools': [], 'experience_terms': []}, 'weight': 0.9},
        'projects': {'keywords': {'hard_skills': [], 'soft_skills': [], 'certifications': [], 'tools': [], 'experience_terms': []}, 'weight': 0.8},
        'education': {'keywords': {'hard_skills': [], 'soft_skills': [], 'certifications': [], 'tools': [], 'experience_terms': []}, 'weight': 0.7},
        'summary': {'keywords': {'hard_skills': [], 'soft_skills': [], 'certifications': [], 'tools': [], 'experience_terms': []}, 'weight': 0.8},
        'other': {'keywords': {'hard_skills': [], 'soft_skills': [], 'certifications': [], 'tools': [], 'experience_terms': []}, 'weight': 0.6}
    }
    
    # Enhanced section patterns
    section_patterns = {
        'skills': r'(?:technical\s+skills?|skills?|competencies|proficiencies|technologies)[\s\S]*?(?=\n[A-Z][a-z]+\s*:|\n\n|\Z)',
        'experience': r'(?:experience|work|employment|career|intern)[\s\S]*?(?=\n[A-Z][a-z]+\s*:|\n\n|\Z)',
        'projects': r'(?:projects?|portfolio)[\s\S]*?(?=\n[A-Z][a-z]+\s*:|\n\n|\Z)',
        'education': r'(?:education|academic|university|degree|b\.tech|bachelor|master)[\s\S]*?(?=\n[A-Z][a-z]+\s*:|\n\n|\Z)',
        'summary': r'(?:objective|summary|profile|about)[\s\S]*?(?=\n[A-Z][a-z]+\s*:|\n\n|\Z)'
    }
    
    # Comprehensive keyword categories
    all_categories = {
        'hard_skills': [
            'python', 'java', 'javascript', 'c++', 'c', 'sql', 'r',
            'machine learning', 'artificial intelligence', 'ai', 'ml', 'deep learning',
            'langchain', 'autogen', 'crewai', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'hugging face', 'transformers', 'llm', 'large language models', 'nlp',
            'natural language processing', 'prompt engineering', 'rag', 'retrieval augmented generation',
            'lstm', 'cnn', 'computer vision', 'data science',
            'azure', 'aws', 'gcp', 'docker', 'kubernetes', 'serverless', 'cloud deployment',
            'microsoft azure', 'app services', 'azure functions', 'cosmosdb',
            'mongodb', 'oracle', 'postgresql', 'mysql', 'database',
            'git', 'github', 'colab', 'jupyter', 'ollama', 'tavily', 'geoapify',
            'big data', 'data structures', 'algorithms', 'analytics', 'optimization'
        ],
        'soft_skills': [
            'problem-solving', 'analytical thinking', 'communication', 'teamwork', 
            'collaboration', 'cross-functional', 'leadership', 'research', 'learning',
            'creative', 'adaptable', 'organized', 'detail-oriented', 'innovative',
            'critical thinking', 'decision-making', 'troubleshooting'
        ],
        'certifications': [
            'bachelor', 'b.tech', 'computer science', 'ai/ml', 'mathematics',
            'machine learning specialization', 'stanford', 'coursera', 'deeplearning.ai',
            'java certification', 'mongodb', 'oracle academy', 'udemy'
        ],
        'tools': [
            'git', 'docker', 'colab', 'jupyter', 'ollama', 'lm studio', 'github',
            'azure portal', 'tavily api', 'geoapify api', 'data guard', 'version control'
        ],
        'experience_terms': [
            'intern', 'experience', 'projects', 'developed', 'built', 'integrated',
            'deployed', 'implemented', 'optimized', 'automated', 'upgraded',
            'fine-tuned', 'worked with', 'applied', 'explored', 'achieved'
        ]
    }
    
    print(f"DEBUG: Analyzing resume sections...")
    
    for section_name, pattern in section_patterns.items():
        section_match = re.search(pattern, resume_text, re.IGNORECASE | re.MULTILINE)
        if section_match:
            section_text = section_match.group()
            print(f"DEBUG: Found {section_name} section with {len(section_text)} characters")
            sections[section_name]['keywords'] = extract_keywords_from_text(section_text, all_categories)
        else:
            print(f"DEBUG: No {section_name} section found, searching entire text")
            # If no specific section found, search entire text with lower weight
            sections[section_name]['keywords'] = extract_keywords_from_text(resume_text, all_categories)
            sections[section_name]['weight'] *= 0.5  # Reduce weight for non-section matches
    
    print(f"DEBUG: Resume section analysis complete")
    return sections


def calculate_keyword_match_score(job_keywords: Dict, resume_analysis: Dict) -> float:
    """Calculate weighted keyword match score"""
    total_score = 0
    max_possible_score = 0
    
    # Count total job keywords
    total_job_keywords = 0
    for keywords in job_keywords.values():
        if isinstance(keywords, list):
            total_job_keywords += len(keywords)
    
    print(f"DEBUG: Total job keywords to match: {total_job_keywords}")
    
    if total_job_keywords == 0:
        return 0
    
    matched_keywords = []
    
    # Calculate weighted matches
    for category, job_kw_list in job_keywords.items():
        if not isinstance(job_kw_list, list):
            continue
            
        print(f"DEBUG: Checking {len(job_kw_list)} keywords in category '{category}'")
            
        for job_keyword in job_kw_list:
            max_possible_score += 1.0  # Each keyword has max score of 1
            keyword_found = False
            
            # Check each resume section for this keyword
            for section_name, section_data in resume_analysis.items():
                if not isinstance(section_data, dict) or 'keywords' not in section_data:
                    continue
                    
                section_keywords = section_data['keywords']
                section_weight = section_data.get('weight', 0.5)
                
                # Handle nested keyword structure
                if isinstance(section_keywords, dict):
                    # Check all categories within this section
                    for resume_category, resume_kw_list in section_keywords.items():
                        if isinstance(resume_kw_list, list) and job_keyword in resume_kw_list:
                            total_score += section_weight
                            matched_keywords.append(f"{job_keyword} (in {section_name}/{resume_category})")
                            keyword_found = True
                            break
                elif isinstance(section_keywords, list):
                    # Old structure: flat list
                    if job_keyword in section_keywords:
                        total_score += section_weight
                        matched_keywords.append(f"{job_keyword} (in {section_name})")
                        keyword_found = True
                        break
                
                if keyword_found:
                    break  # Don't double count across sections
    
    print(f"DEBUG: Matched keywords ({len(matched_keywords)}): {matched_keywords[:10]}...")  # Show first 10
    print(f"DEBUG: Total score: {total_score}, Max possible: {max_possible_score}")
    
    # Convert to percentage
    if max_possible_score > 0:
        percentage = min(100, (total_score / max_possible_score) * 100)
        print(f"DEBUG: Final keyword match percentage: {percentage:.2f}%")
        return percentage
    return 0


def find_missing_keywords(job_keywords: Dict, resume_analysis: Dict) -> List[str]:
    """Find keywords present in job description but missing from resume"""
    all_job_keywords = []
    for category, keywords in job_keywords.items():
        if isinstance(keywords, list):
            all_job_keywords.extend(keywords)
    
    all_resume_keywords = []
    for section_data in resume_analysis.values():
        if isinstance(section_data, dict) and 'keywords' in section_data:
            section_keywords = section_data['keywords']
            if isinstance(section_keywords, dict):
                for category, keywords in section_keywords.items():
                    if isinstance(keywords, list):
                        all_resume_keywords.extend(keywords)
            elif isinstance(section_keywords, list):
                all_resume_keywords.extend(section_keywords)
    
    missing = list(set(all_job_keywords) - set(all_resume_keywords))
    print(f"DEBUG: All job keywords: {len(all_job_keywords)}, All resume keywords: {len(all_resume_keywords)}, Missing: {len(missing)}")
    return missing


#add similarity from faiss query results
@tool("Confidence_score_calculation_tool")
def confidence_score_calculation_tool(similarity: float, keyword_match_score: float = None, 
                                    experience_match: float = None, ats_readability: float = None) -> float:
    """
    Calculate ATS confidence score based on multiple weighted factors as per industry standards.
    
    Final Score Calculation (from README):
    - Keyword match score (50–70% weight)
    - Experience & qualification match (20–30% weight)  
    - Formatting & ATS-readability score (10–20% weight)
    
    Args:
        similarity (float): Base FAISS similarity percentage (0-100)
        keyword_match_score (float): Weighted keyword matching score (0-100)
        experience_match (float): Experience and qualification match (0-100)
        ats_readability (float): ATS readability score (0-100)
        
    Returns:
        float: The weighted ATS confidence score (0-100)
    """
    if not (0 <= similarity <= 100):
        raise ValueError("Similarity must be between 0 and 100.")
    
    # Use provided scores or defaults based on similarity
    keyword_score = keyword_match_score if keyword_match_score is not None else similarity
    experience_score = experience_match if experience_match is not None else similarity * 0.9
    readability_score = ats_readability if ats_readability is not None else 85.0  # Assume good format
    
    # ATS Industry Standard Weights
    weights = {
        'keyword_match': 0.60,      # 60% weight - primary factor
        'experience_match': 0.25,   # 25% weight - qualification match
        'ats_readability': 0.15     # 15% weight - formatting
    }
    
    # Calculate weighted score
    weighted_score = (
        keyword_score * weights['keyword_match'] +
        experience_score * weights['experience_match'] +
        readability_score * weights['ats_readability']
    )
    
    # Apply industry-standard adjustments
    
    # 1. Keyword threshold penalty - heavy penalty if keyword match is too low
    if keyword_score < 40:
        weighted_score *= 0.7  # 30% penalty for poor keyword match
    
    # 2. Experience gap penalty
    if experience_score < 50:
        weighted_score *= 0.85  # 15% penalty for experience mismatch
    
    # 3. Minimum viable candidate threshold
    if weighted_score < 30:
        weighted_score *= 0.8  # Additional penalty for very low scores
    
    # 4. Excellence bonus for high performers
    if weighted_score > 85 and keyword_score > 80:
        weighted_score = min(100, weighted_score * 1.05)  # 5% bonus for top candidates
    
    # Ensure bounds and round
    final_score = np.clip(weighted_score, 0, 100)
    
    return np.round(final_score, 2)


@tool("Hiring_decision_tool")
def hiring_decision_tool(confidence: float, threshold: float=75) -> str:
    """
    Make a hiring decision based on the confidence score and a threshold.

    Args:
        confidence (float): The confidence score of the candidate.
        threshold (float, optional): The threshold for making a positive hiring decision. Defaults to 75.

    Returns:
        str: The hiring decision ("Hire" or "No Hire").
    """
    if not (0 <= confidence <= 100):
        raise ValueError("Confidence must be between 0 and 100.")

    if confidence >= threshold:
        return "Hire"
    else:
        return "No Hire"