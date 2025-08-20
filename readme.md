# Resume ATS (Applicant Tracking System) 
## Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: LangChain, Google Gemini AI, FAISS Vector Store
- **Data Processing**: PyMuPDF, Pydantic (structured data validation)
- **Analysis**
## How It Works

1. **Document Processing**: Advanced extraction fro## 🔮 Future Enhancements

- **Advanced Analytics Dashboard**: Comprehensive reporting and trend analysis
- **Bulk Resume Processing**: Batch analysis capabilities for HR departments
- **Custom Scoring Weights**: Configurable industry-specific scoring parameters
- **Integration APIs**: Connectors for popular ATS platforms (Workday, Greenhouse, etc.)
- **Multi-Language Support**: Analysis in multiple languages with localized keywords
- **Resume Builder Integration**: AI-powered resume optimization suggestions
- **Video Resume Analysis**: Transcription and soft skills assessment
- **Bias Detection**: Algorithmic fairness monitoring and reporting
- **Advanced Document Formats**: Support for DOCX, LinkedIn profiles, and web portfolios
- **Real-time Collaboration**: Multi-user analysis and commenting system

## Advanced Features in Current Version

### **Keyword Analysis Engine**
- **5000+ Industry Keywords**: Comprehensive database across tech, business, and soft skills
- **Fuzzy Matching**: Handles abbreviations, variations, and compound terms
- **Section-Weight Optimization**: Strategic placement scoring for ATS optimization
- **Context Awareness**: Evaluates keyword usage context and relevance

### **Scoring Algorithm Details**
- **Multi-Factor Weighting**: Configurable industry-standard weights
- **Penalty System**: Intelligent score adjustments based on performance thresholds
- **Bonus Mechanisms**: Rewards for exceptional keyword matching and experience alignment
- **Normalized Scaling**: Ensures consistent 0-100% scoring regardless of input variation

### **Data Extraction & Structuring**
- **Pydantic Validation**: Type-safe data extraction with automatic error handling
- **Entity Recognition**: Advanced extraction of contact info, experience, education
- **Metadata Preservation**: Maintains document context and analysis history
- **JSON Schema Compliance**: Standardized output format for integrationsumes and job descriptions with metadata preservation
2. **Structured Analysis**: Pydantic-based parsing into standardized schemas for consistent data handling
3. **Multi-Vector Similarity**: FAISS semantic matching combined with keyword-based analysis
4. **Weighted Scoring**: Industry-standard ATS scoring with configurable weights and penalty modifiers
5. **Entity Indexing**: Automatic extraction and storage of skills, tools, and certifications for improved matching
6. **Decision Logic**: Threshold-based hiring recommendations with confidence intervals*: 
  - Advanced ATS keyword analyzer with 5 category classification
  - Weighted confidence scoring algorithm
  - Semantic similarity matching with distance optimization
- **Frontend**: HTML, CSS, JavaScript
- **Environment**: Python Virtual Environment

## How the Advanced ATS Analysis Works

### **1. Structured Data Extraction**
- **Resume Parsing**: Extracts contact info, work experience, education, skills, and certifications
- **Job Description Parsing**: Identifies required skills, experience, education, and industry keywords
- **Entity Classification**: Categorizes elements into hard skills, soft skills, tools, and certifications

### **2. Multi-Dimensional Keyword Analysis**
- **5-Category Classification**: Hard skills, soft skills, certifications, tools, experience terms
- **Fuzzy Matching**: Handles variations and abbreviations (ML vs Machine Learning)
- **Section-Weighted Scoring**: Skills section (1.0x), Experience (0.9x), Projects (0.8x), etc.
- **Context Relevance**: Evaluates keyword placement and context

### **3. Advanced Scoring Algorithm**
```
Final Score = (Keyword_Match × 0.60) + (Experience_Match × 0.25) + (ATS_Readability × 0.15)

With penalty adjustments:
- Keyword score < 40%: Apply 0.7x modifier
- Experience score < 50%: Apply 0.85x modifier  
- Overall score < 30%: Apply 0.8x modifier
- Excellent performance (>85% + keyword >80%): Apply 1.05x bonus
```

### **4. Intelligent Vector Store Management**
- **Auto-Indexing**: Automatically adds analyzed resumes and job descriptions
- **Entity Extraction**: Indexes individual skills, tools, and certifications
- **Optimized Similarity**: Uses `1/(1+distance)` mapping for stable percentage scores
- **Dynamic Knowledge Base**: Continuously improves matching accuracy
An advanced AI-powered web application that provides industry-standard ATS analysis of resumes against job descriptions using LangChain, Google's Gemini AI, FAISS vector similarity search, and comprehensive keyword analysis. The system delivers professional-grade evaluation with weighted scoring, structured data extraction, and detailed recommendations.

## Features

### **Core ATS Analysis**
- **Advanced Keyword Analysis**: Industry-standard ATS keyword matching with weighted scoring across multiple categories
- **Structured Data Extraction**: Professional parsing of both resumes and job descriptions using Pydantic schemas
- **Multi-Factor Scoring**: Weighted confidence scoring based on:
  - Keyword Matching (60% weight)
  - Experience Relevance (25% weight) 
  - ATS Readability (15% weight)
- **FAISS Vector Search**: Semantic similarity matching with optimized distance-to-percentage conversion
- **Entity Indexing**: Automatic extraction and indexing of skills, certifications, and tools

### **Professional Analysis Components**
- **Resume Analysis**: Upload PDF resumes and get comprehensive AI-powered analysis
- **Job Description Matching**: Compare resumes against job descriptions (text or PDF)
- **Missing Skills Detection**: Identifies gaps in candidate qualifications with categorization
- **Hiring Recommendations**: AI-driven hiring decisions with industry-standard thresholds
- **Job Recommendations**: Suggests relevant positions based on resume content
- **ATS Readability Score**: Evaluates resume format for ATS parsing compatibility

### **Technical Features**
- **Robust JSON Parsing**: Pydantic-based structured output with fallback mechanisms
- **Dynamic Vector Store**: Auto-expanding knowledge base with resume and job description indexing
- **API Endpoints**: RESTful API for programmatic integration
- **Web Interface**: Clean, responsive Flask web application(Applicant Tracking System) Tracker

An AI-powered web application that analyzes resumes against job descriptions using LangChain, Google's Gemini AI, and FAISS vector similarity search. The system provides match scores, identifies missing skills, and offers hiring recommendations.

## Features

- **Resume Analysis**: Upload PDF resumes and get AI-powered analysis
- **Job Description Matching**: Compare resumes against job descriptions (text or PDF)
- **FAISS Vector Search**: Uses vector similarity for accurate matching
- **Missing Skills Detection**: Identifies gaps in candidate qualifications
- **Hiring Recommendations**: AI-driven hiring decisions with confidence scores
- **Job Recommendations**: Suggests relevant positions based on resume content
- **Web Interface**: Clean, responsive Flask web application

## Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: LangChain, Google Gemini AI, FAISS
- **Document Processing**: PyMuPDF
- **Frontend**: HTML, CSS, JavaScript
- **Environment**: Python Virtual Environment

## Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini AI)
- Git (optional, for cloning)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://www.github.com/degenerate-kun-69/RESUME-ATS-TRACKER.git
cd RESUME-ATS-TRACKER
```

### 2. Create Virtual Environment

```bash
python3 -m venv env #python -m venv env for windows
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root & add your Google API key to the `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Prepare Vector Store

The application now includes an **auto-optimized vector store** that creates itself on first run. However, you can enhance it by adding your own data:

**Option A: Use Auto-Generated Store (Recommended)**
- The system automatically creates an ATS-optimized vector store with industry-standard patterns
- No additional setup required

**Option B: Add Custom Data**
```python
from llm.langchain_setup import add_resume_to_vector_store, add_job_description_to_vector_store

# Add individual resumes
add_resume_to_vector_store(
    resume_text="Your resume content here...",
    metadata={"candidate_id": "123", "position": "Software Engineer"}
)

# Add job descriptions
add_job_description_to_vector_store(
    job_desc_text="Your job description here...",
    metadata={"job_id": "456", "company": "TechCorp"}
)
```

### 6. Run the Application

```bash
python3 app.py
```

The application will start on `http://127.0.0.1:5000`

## Usage

### Web Interface

1. **Access the Application**: Open `http://127.0.0.1:5000` in your browser

2. **Job Description Input**:
   - **Text Input**: Paste job description directly
   - **PDF Upload**: Upload job description as PDF file

3. **Resume Upload**: Upload candidate's resume in PDF format

4. **Analysis**: Click "Analyze Resume" to get comprehensive results:
   - **Match Scores**: Multiple percentage scores for different aspects
   - **Confidence Score**: Industry-weighted final score
   - **Hiring Decision**: AI recommendation with threshold-based logic
   - **Missing Skills**: Categorized list of gaps to address
   - **Profile Summary**: AI-generated candidate assessment
   - **Detailed Breakdown**: Score components and structured data
   - **Recommendations**: Actionable improvement suggestions

### API Endpoints

The application provides comprehensive API endpoints for programmatic integration:

**Resume Analysis**
- `POST /api/analyze` - Full resume analysis with structured output
  ```json
  {
    "resume_text": "Your resume content...",
    "job_description": "Job requirements..."
  }
  ```

**Health Check**
- `GET /health` - Application health status

**API Response Structure**
```json
{
  "success": true,
  "analysis": {
    "Match": "78.5%",
    "Confidence Score": 82.3,
    "Hiring Decision": "Hire",
    "Missing Keywords and Skills": ["kubernetes", "docker"],
    "Profile Summary": "Strong candidate with relevant experience...",
    "Score Breakdown": {
      "Keyword Matching (60% weight)": "78.5%",
      "Experience Match (25% weight)": "85.0%",
      "ATS Readability (15% weight)": "90.0%"
    },
    "Detailed Analysis": { /* Structured data */ }
  }
}
```

## Project Structure

```
langchain-resume-ats/
├── app.py                 # Flask application entry point
├── config.py             # Configuration and environment variables
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── vector_store/         # FAISS vector store (create this)
├── instance/
│   └── temp/            # Temporary file storage
├── llm/
│   ├── langchain_setup.py  # Advanced ATS engine with Pydantic schemas
│   └── tools.py           # Keyword analyzer and scoring tools
├── routes/
│   ├── main_routes.py     # Web interface routes
│   └── api_routes.py      # API endpoints
├── services/
│   ├── classification.py  # Resume classification orchestrator
│   ├── extraction.py      # PDF text extraction utilities
│   ├── parser.py         # JSON response parsing with fallbacks
│   └── recommender.py    # Job recommendation engine
└── templates/
    └── index.html        # Web interface template
```

## How It Works

1. **Document Processing**: Extracts text from PDF resumes and job descriptions
2. **Vector Similarity**: Uses FAISS to calculate similarity between resume and job description
3. **AI Analysis**: Gemini AI analyzes qualitative aspects and identifies missing skills
4. **Scoring**: Combines vector similarity with AI confidence scoring
5. **Decision Making**: Provides hiring recommendations based on threshold scoring

## Troubleshooting

### Common Issues

**1. "No module named 'google.generativeai'"**
```bash
pip install google-generativeai
```

**2. "vector_store not found"**
- The system now auto-creates an optimized vector store on first run
- If issues persist, delete the `vector_store` directory and restart the application

**3. "Confidence score and match are identical"**
- **Fixed**: Now uses weighted multi-factor scoring algorithm
- Match shows keyword similarity, Confidence shows weighted final score
- Different scores indicate proper ATS-standard evaluation

**4. "Missing keywords not properly categorized"**
- The enhanced analyzer now categorizes missing skills by type
- Provides specific recommendations for each category

**3. "GOOGLE_API_KEY not found"**
- Verify your `.env` file exists and contains the API key
- Ensure the key is valid and has proper permissions

**4. PDF processing errors**
- Ensure uploaded PDFs are not corrupted or password-protected
- Check that PyMuPDF is properly installed
- The system now handles OCR-scanned PDFs better

**5. "JSON parsing errors"**
- **Fixed**: Robust Pydantic-based parsing with multiple fallback strategies
- System handles malformed LLM responses gracefully
- Structured output guaranteed even with API failures

### Performance Tips

- **Vector Store Optimization**: The auto-generated store improves with each analysis
- **API Efficiency**: Batch multiple analyses to leverage cached embeddings
- **Keyword Matching**: System now handles 1000+ industry-standard keywords and variations
- **Memory Management**: Entity indexing is optimized for production use
- **Response Time**: Pydantic schemas reduce parsing overhead significantly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

## Future Enhancements

- Support for additional document formats (DOCX, TXT)
- Batch processing capabilities
- Enhanced job recommendation algorithms
- Integration with job boards APIs
- Resume parsing for structured data extraction
- Multi-language support