from flask import Blueprint, request, jsonify
from services.classification import classify_resume
from services.parser import parse_json_response

api = Blueprint('api', __name__)

@api.route('/api/analyze', methods=['POST'])
def api_analyze_resume():
    data = request.get_json()
    if not data or 'resume_text' not in data or 'job_description' not in data:
        return jsonify({"error": "Missing resume_text or job_description"}), 400
    try:
        result = classify_resume(data['resume_text'], data['job_description'])
        parsed = parse_json_response(result)
        return jsonify({"success": True, "analysis": parsed})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/health')
def health_check():
    return jsonify({"status": "healthy"})
