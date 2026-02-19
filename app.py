from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from routes.main_routes import main as main_blueprint
from routes.api_routes import api as api_blueprint
from utils.redis_cache import init_redis
import os

# Initialize rate limiter (will be set up in create_app)
limiter = None

def create_app():
    global limiter
    
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = './instance/temp'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize Redis
    redis_client = init_redis()
    
    # Setup rate limiting
    if redis_client:
        # Use Redis for rate limiting if available
        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=["200 per day", "50 per hour"],
            storage_uri=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
        )
    else:
        # Fallback to memory-based rate limiting
        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=["200 per day", "50 per hour"],
            storage_uri="memory://"
        )
    
    # Register Blueprints
    app.register_blueprint(main_blueprint)
    app.register_blueprint(api_blueprint)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
