from flask import Flask
from extensions import limiter
from routes.main_routes import main as main_blueprint
from routes.api_routes import api as api_blueprint
from utils.redis_cache import init_redis
import os


def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = './instance/temp'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize Redis
    redis_client = init_redis()

    # Configure rate limiter storage via app config
    if redis_client:
        app.config['RATELIMIT_STORAGE_URI'] = (
            f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
        )
    else:
        app.config['RATELIMIT_STORAGE_URI'] = "memory://"

    limiter.init_app(app)

    # Register Blueprints
    app.register_blueprint(main_blueprint)
    app.register_blueprint(api_blueprint)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
