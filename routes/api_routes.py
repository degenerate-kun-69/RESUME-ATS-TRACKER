from flask import Blueprint, request, jsonify
from services.classification import classify_resume_async
from services.parser import parse_json_response

api = Blueprint('api', __name__)

from extensions import limiter

@api.route('/api/analyze', methods=['POST'])
@limiter.limit("10 per minute")
async def api_analyze_resume():
    """Async API endpoint for resume analysis with rate limiting"""
    data = request.get_json()
    if not data or 'resume_text' not in data or 'job_description' not in data:
        return jsonify({"error": "Missing resume_text or job_description"}), 400
    try:
        result = await classify_resume_async(data['resume_text'], data['job_description'])
        parsed = parse_json_response(result) if isinstance(result, str) else result
        return jsonify({"success": True, "analysis": parsed})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/health')
async def health_check():
    """Health check endpoint"""
    from utils.redis_cache import get_cache_stats
    cache_stats = get_cache_stats()
    return jsonify({
        "status": "healthy",
        "cache": cache_stats
    })

@api.route('/api/cache/clear', methods=['POST'])
@limiter.limit("5 per hour")
async def clear_cache():
    """Clear cache endpoint (rate limited)"""
    from utils.redis_cache import clear_cache_pattern
    try:
        cleared = clear_cache_pattern("resume_analysis:*")
        return jsonify({"success": True, "cleared_keys": cleared})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
