from flask import Blueprint, request, render_template, redirect, url_for, current_app, Response, stream_with_context
import os
import json as _json
import queue as _queue
import threading as _thread
from services.extraction import extract_text_from_pdf
from services.classification import classify_resume_async
from services.parser import parse_json_response
from services.recommender import generate_job_recommendations_async, generate_job_recommendations
from extensions import limiter

main = Blueprint('main', __name__)

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


@main.route('/analyze/stream', methods=['POST'])
@limiter.limit("5 per minute")
def analyze_resume_stream():
    """SSE endpoint (Issue 10): emits live progress events while
    evaluate_resume() runs in a background thread."""

    if 'resume' not in request.files:
        return _json.dumps({"error": "No resume file"}), 400

    resume_file = request.files['resume']
    job_input_type = request.form.get('job_input_type', 'text')

    if resume_file.filename == '':
        return _json.dumps({"error": "No resume selected"}), 400

    # --- Resolve job description ---
    if job_input_type == 'text':
        if 'job_description' not in request.form or not request.form['job_description'].strip():
            return _json.dumps({"error": "Please provide a job description"}), 400
        job_description = request.form['job_description']
    elif job_input_type == 'file':
        if 'job_description_file' not in request.files:
            return _json.dumps({"error": "Please upload a job description file"}), 400
        job_desc_file = request.files['job_description_file']
        if job_desc_file.filename == '':
            return _json.dumps({"error": "Please select a job description file"}), 400
        job_desc_path = os.path.join(
            current_app.config['UPLOAD_FOLDER'], f"job_desc_{job_desc_file.filename}"
        )
        job_desc_file.save(job_desc_path)
        try:
            job_description = extract_text_from_pdf(job_desc_path)
            if not job_description.strip():
                return _json.dumps({"error": "Could not extract text from job description PDF"}), 400
        except Exception as e:
            return _json.dumps({"error": f"Error reading job description PDF: {str(e)}"}), 500
        finally:
            if os.path.exists(job_desc_path):
                os.remove(job_desc_path)
    else:
        return _json.dumps({"error": "Invalid job input type"}), 400

    # --- Extract resume text before streaming begins ---
    resume_path = os.path.join(current_app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_path)
    try:
        resume_text = extract_text_from_pdf(resume_path)
    except Exception as e:
        return _json.dumps({"error": f"Could not read resume PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(resume_path):
            os.remove(resume_path)

    # --- Background thread sends events through a queue ---
    progress_q = _queue.Queue()

    def _progress(step, message, percent):
        progress_q.put(('progress', {'step': step, 'message': message, 'percent': percent}))

    def _run():
        try:
            from llm.langchain_setup import evaluate_resume
            result = evaluate_resume(resume_text, job_description, progress_callback=_progress)
            _progress("recommendations", "Generating job recommendations...", 95)
            recommendations = generate_job_recommendations(job_description, resume_text)
            progress_q.put(('result', {'result': result, 'recommendations': recommendations}))
        except Exception as exc:
            progress_q.put(('error', {'error': str(exc)}))
        finally:
            progress_q.put(None)  # sentinel

    worker = _thread.Thread(target=_run, daemon=True)
    worker.start()

    def generate():
        yield f"event: progress\ndata: {_json.dumps({'step': 'start', 'message': 'Starting analysis...', 'percent': 5})}\n\n"
        while True:
            try:
                item = progress_q.get(timeout=180)
            except _queue.Empty:
                yield f"event: error\ndata: {_json.dumps({'error': 'Analysis timed out after 3 minutes'})}\n\n"
                break
            if item is None:
                break
            event_type, data = item
            yield f"event: {event_type}\ndata: {_json.dumps(data)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )
