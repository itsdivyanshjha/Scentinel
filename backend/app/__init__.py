import os
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import urllib.parse

mongo = PyMongo()
jwt = JWTManager()

def create_app(test_config=None):
    # Load environment variables
    load_dotenv()
    
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # MongoDB connection string with new credentials
    username = "divyanshapp"
    password = "Divyansh@2025"
    mongodb_uri = f"mongodb+srv://{username}:{urllib.parse.quote_plus(password)}@scentinelcluster.apxy5nv.mongodb.net/scentinel?retryWrites=true&w=majority"
    
    # Set up configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        MONGO_URI=mongodb_uri,
        JWT_SECRET_KEY=os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key'),
    )
    
    # Enable CORS
    CORS(app)
    
    # Initialize extensions
    mongo.init_app(app)
    jwt.init_app(app)
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.perfumes import perfumes_bp
    from app.routes.recommend import recommend_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(perfumes_bp)
    app.register_blueprint(recommend_bp)
    
    # Add root route
    @app.route('/')
    def home():
        return {
            'message': 'Scentinel Perfume Recommendation API',
            'status': 'running',
            'endpoints': {
                'health': '/health',
                'perfume_count': '/perfumes/count',
                'auth': '/auth/',
                'perfumes': '/perfumes/',
                'recommendations': '/recommend/'
            }
        }, 200
    
    # Add health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'mongodb': 'connected', 'port': '5001'}, 200
    
    # Add simple test endpoint  
    @app.route('/perfumes/count')
    def perfume_count():
        count = mongo.db.perfumes.count_documents({})
        return {'perfume_count': count}, 200
    
    return app 