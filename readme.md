# Resume ATS (Applicant Tracking System) Tracker

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

**Important**: You need to create a FAISS vector store with resume data before running the application.

The current setup expects a vector store named `vector_store` in the project root. You'll need to:

1. Collect resume samples (PDF format)
2. Create a script to process resumes and build the FAISS index
3. Save the vector store to the `vector_store` directory

Example script to create vector store:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
import os

# Initialize embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load and process resume documents
documents = []
resume_folder = "./temp/<pdf-name>"

for filename in os.listdir(resume_folder):
    if filename.endswith('.pdf'):
        loader = PyMuPDFLoader(os.path.join(resume_folder, filename))
        docs = loader.load()
        documents.extend(docs)

# Create and save vector store
vector_store = FAISS.from_documents(documents, embedding_model)
vector_store.save_local("vector_store")
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

4. **Analysis**: Click "Analyze Resume" to get:
   - Match percentage
   - Confidence score
   - Hiring recommendation
   - Missing skills
   - Profile summary
   - Job recommendations

### API Endpoints

The application also provides API endpoints (check `routes/api_routes.py` for implementation):

- `POST /api/analyze` - Programmatic resume analysis

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
│   ├── langchain_setup.py  # LangChain configuration
│   └── tools.py           # Custom tools for scoring
├── routes/
│   ├── main_routes.py     # Web interface routes
│   └── api_routes.py      # API routes
├── services/
│   ├── classification.py  # Resume classification logic
│   ├── extraction.py      # PDF text extraction
│   ├── parser.py         # Response parsing utilities
│   └── recommender.py    # Job recommendation logic
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
- Ensure you've created the FAISS vector store as described in step 5
- Check that the `vector_store` directory exists in project root

**3. "GOOGLE_API_KEY not found"**
- Verify your `.env` file exists and contains the API key
- Ensure the key is valid and has proper permissions

**4. PDF processing errors**
- Ensure uploaded PDFs are not corrupted
- Check that PyMuPDF is properly installed

### Performance Tips

- **Vector Store**: Use high-quality, diverse resume samples for better matching
- **API Limits**: Monitor Google API usage to avoid rate limits
- **File Size**: Keep PDF uploads under 10MB for optimal performance

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