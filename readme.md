# Advanced Resume ATS Tracker

A sophisticated AI-powered Applicant Tracking System (ATS) that provides professional-grade resume analysis using cutting-edge machine learning technologies. Built with LangChain, Google Gemini AI, and FAISS vector search to deliver industry-standard recruitment insights.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.27-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Key Features

### **Advanced ATS Analysis**
- **Multi-Factor Scoring**: Weighted evaluation using industry-standard ATS criteria
- **Semantic Similarity**: FAISS vector search for deep content understanding  
- **Keyword Intelligence**: 5000+ categorized keywords with fuzzy matching
- **Structured Data Extraction**: Pydantic-validated parsing of resumes and job descriptions

### **AI-Powered Insights**
- **Google Gemini Integration**: State-of-the-art language model analysis
- **Section-Weighted Scoring**: Strategic evaluation of resume sections
- **Experience Matching**: Relevance assessment for career progression
- **ATS Readability**: Format optimization recommendations

### **Professional Scoring System**
```
Final Score = Keyword Match (60%) + Experience Match (25%) + ATS Readability (15%)
```
- **Penalty System**: Intelligent adjustments for low performance areas
- **Bonus Rewards**: Recognition for exceptional keyword alignment
- **Threshold-Based Decisions**: Configurable hiring recommendations

### **Self-Improving Architecture**
- **Dynamic Vector Store**: Auto-expanding knowledge base
- **Entity Indexing**: Continuous learning from analyzed documents
- **Performance Optimization**: Cached embeddings and smart retrieval

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Flask 3.1.1 | Web application framework |
| **AI Engine** | Google Gemini 2.5-Pro | Advanced language processing |
| **Vector Search** | FAISS | Semantic similarity matching |
| **Data Validation** | Pydantic 2.11.7 | Type-safe structured parsing |
| **Document Processing** | PyMuPDF 1.26.3 | PDF text extraction |
| **Orchestration** | LangChain 0.3.27 | AI workflow management |
| **Frontend** | HTML5/CSS3/JavaScript | Responsive web interface |

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Google API key for Gemini AI
- 4GB+ RAM (recommended for vector operations)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/degenerate-kun-69/RESUME-ATS-TRACKER.git
cd RESUME-ATS-TRACKER
```

2. **Set up virtual environment**
```bash
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

5. **Launch application**
```bash
python app.py
```

Access the application at `http://localhost:5000`

## Usage Guide

### Web Interface
1. **Upload Resume**: PDF format recommended for best parsing
2. **Job Description**: Paste text or upload PDF
3. **Analyze**: Get comprehensive ATS evaluation
4. **Review Results**: Detailed scoring breakdown and recommendations

### API Integration
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Your resume content...",
    "job_description": "Job requirements..."
  }'
```

## Project Architecture

```
resume-ats-tracker/
├── app.py                     # Flask application entry point
├── config.py                  # Environment configuration
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create this)
│
├── llm/                       # AI Engine Core
│   ├── langchain_setup.py     # Advanced ATS analysis engine
│   └── tools.py              # Keyword analyzer & scoring tools
│
├── routes/                    # Web & API Routes
│   ├── main_routes.py         # Web interface endpoints
│   └── api_routes.py          # RESTful API endpoints
│
├── services/                  # Business Logic
│   ├── classification.py     # Resume classification
│   ├── extraction.py         # PDF text extraction
│   ├── parser.py             # Response parsing
│   └── recommender.py        # Job recommendations
│
├── templates/                 # Web Interface
│   └── index.html            # Main application UI
│
├── instance/                  # Runtime Data
│   └── temp/                 # Temporary file storage
│
└── vector_store/             # AI Knowledge Base
    ├── index.faiss          # Vector embeddings
    └── index.pkl            # Metadata store
```

## How It Works

### 1. Document Processing Pipeline
```python
# PDF text extraction with metadata preservation
resume_text = extract_text_from_pdf(file_path)
job_desc_text = parse_job_description(input_data)
```

### 2. Multi-Dimensional Analysis
```python
# Structured data extraction using Pydantic schemas
parsed_data = classifier_chain.invoke({
    "resume": resume_text,
    "job_description": job_description
})

# Advanced keyword analysis with 5 categories
keyword_analysis = ats_keyword_analyzer.invoke({
    "resume_text": resume_text,
    "job_description": job_description
})
```

### 3. Intelligent Scoring Algorithm
```python
# Weighted confidence calculation
confidence_score = (
    keyword_match_score * 0.60 +      # Primary ATS factor
    experience_match * 0.25 +         # Career relevance
    ats_readability * 0.15            # Format optimization
)
```

### 4. Vector Similarity Matching
```python
# Semantic search with optimized distance conversion
similarity_results = vector_store.similarity_search_with_score(query, k=1)
similarity_percentage = 1.0 / (1.0 + distance) * 100
```

## Analysis Output

### Comprehensive Scoring Report
```json
{
  "Match": "78.5%",
  "Confidence Score": 82.3,
  "Hiring Decision": "Hire",
  "Missing Keywords": ["kubernetes", "docker", "microservices"],
  "Profile Summary": "Strong full-stack developer with relevant experience...",
  "Score Breakdown": {
    "Keyword Matching (60%)": "78.5%",
    "Experience Match (25%)": "85.0%", 
    "ATS Readability (15%)": "90.0%"
  },
  "Detailed Analysis": {
    "hard_skills": ["python", "javascript", "react"],
    "soft_skills": ["teamwork", "problem-solving"],
    "certifications": ["aws certified"],
    "experience_years": "3-5 years"
  }
}
```

### Visual Feedback
- **Match Percentage**: Direct keyword alignment
- **Confidence Score**: Weighted final evaluation
- **Hiring Recommendation**: Threshold-based decision
- **Improvement Areas**: Specific skill gaps identified
- **ATS Optimization**: Format and structure suggestions

## Advanced Features

### Keyword Analysis Engine
- **5000+ Industry Terms**: Comprehensive technology and business vocabulary
- **Fuzzy Matching**: Handles variations (ML ↔ Machine Learning)
- **Context Awareness**: Evaluates keyword placement relevance
- **Section Weighting**: Skills (1.0x), Experience (0.9x), Projects (0.8x)

### Dynamic Learning System
- **Auto-Indexing**: Builds knowledge from each analysis
- **Entity Extraction**: Catalogs skills, tools, and certifications
- **Pattern Recognition**: Improves matching accuracy over time
- **Vector Store Growth**: Expanding semantic understanding

### Professional Data Handling
- **Type Safety**: Pydantic validation for all data structures
- **Error Recovery**: Graceful handling of malformed inputs
- **Structured Output**: Consistent JSON schema for integrations
- **Metadata Tracking**: Analysis history and context preservation

## Troubleshooting

### Common Issues

**API Key Configuration**
```bash
# Verify .env file exists and contains valid key
cat .env
export GOOGLE_API_KEY="your_key_here"
```

**Vector Store Initialization**
```bash
# Clear and rebuild vector store if corrupted
rm -rf vector_store/
python app.py  # Will auto-create optimized store
```

**PDF Processing Errors**
- Ensure PDFs are not password-protected
- Check file size (recommended < 10MB)
- Verify PyMuPDF installation: `pip install pymupdf`

**Memory Issues**
- Increase system RAM allocation
- Use smaller batch sizes for bulk processing
- Clear vector store cache periodically

## Configuration

### Environment Variables
```bash
# Required
GOOGLE_API_KEY=your_google_api_key

# Optional
UPLOAD_FOLDER=./instance/temp
DEBUG_MODE=True
MAX_FILE_SIZE=10485760  # 10MB
```

### Scoring Weights (Customizable)
```python
SCORING_WEIGHTS = {
    'keyword_match': 0.60,    # Primary ATS factor
    'experience_match': 0.25, # Career relevance  
    'ats_readability': 0.15   # Format quality
}
```

## 🚀 Performance Optimization

### Best Practices
- **Batch Processing**: Group multiple analyses for efficiency
- **Caching**: Leverage embedded vector cache for repeated queries
- **Resource Management**: Monitor memory usage with large document sets
- **API Limits**: Implement rate limiting for Google Gemini calls

### Production Deployment
- Use WSGI server (Gunicorn) instead of Flask dev server
- Configure reverse proxy (Nginx) for static file serving
- Implement proper logging and monitoring
- Set up automated vector store backups

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black . && isort .
```

## License

This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE) file for details.

## Roadmap

### Upcoming Features
- [ ] **Bulk Processing**: Batch resume analysis for HR departments
- [ ] **Custom Weights**: Industry-specific scoring configurations
- [ ] **Advanced Analytics**: Comprehensive reporting dashboard
- [ ] **API Integrations**: Connectors for major ATS platforms
- [ ] **Multi-Language**: Support for international job markets

### Long-term Vision
- [ ] **Video Resume Analysis**: AI-powered soft skills assessment
- [ ] **Bias Detection**: Algorithmic fairness monitoring
- [ ] **Real-time Collaboration**: Multi-user analysis workflows
- [ ] **Mobile Application**: Native iOS/Android apps
- [ ] **Enterprise Features**: SSO, audit logs, compliance tools

## 📞 Support

- **Documentation**: [Wiki](https://github.com/degenerate-kun-69/RESUME-ATS-TRACKER/wiki)
- **Issues**: [GitHub Issues](https://github.com/degenerate-kun-69/RESUME-ATS-TRACKER/issues)
- **Discussions**: [GitHub Discussions](https://github.com/degenerate-kun-69/RESUME-ATS-TRACKER/discussions)

---