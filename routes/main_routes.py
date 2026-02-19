from flask import Blueprint, request, render_template, redirect, url_for, Flask, current_app
import os
from config import UPLOAD_FOLDER
from services.extraction import extract_text_from_pdf
from services.classification import classify_resume_async
from services.parser import parse_json_response
from services.recommender import generate_job_recommendations_async
from llm.langchain_setup import classifier_chain
import asyncio

main = Blueprint('main', __name__)

# Import limiter from app
from app import limiter

@main.route('/')
async def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
@limiter.limit("5 per minute")
async def analyze_resume():
    # Same logic as your main route, refactored to call service functions
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
        job_desc_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"job_desc_{job_desc_file.filename}")
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
    resume_path = os.path.join(current_app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_path)

    try:
        # Extract resume content using LangChain method
        resume_text = extract_text_from_pdf(resume_path)

        # Run classification using our async service
        classification_result = await classify_resume_async(resume_text, job_description)

        # Generate job recommendations based on the actual job description
        recommendations = await generate_job_recommendations_async(job_description, resume_text)

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
