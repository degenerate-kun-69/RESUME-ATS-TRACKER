from flask import Flask
from routes.main_routes import main as main_blueprint
from routes.api_routes import api as api_blueprint
import os

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = './instance/temp'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register Blueprints
    app.register_blueprint(main_blueprint)
    app.register_blueprint(api_blueprint)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
