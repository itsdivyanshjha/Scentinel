from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import mongo
from app.services.recommendation_service import generate_recommendations
from bson.json_util import dumps
import json

recommend_bp = Blueprint('recommend', __name__, url_prefix='/api/recommend')

@recommend_bp.route('', methods=['GET'])
@jwt_required()
def get_recommendations():
    """Generate and return personalized perfume recommendations based on user rankings"""
    user_id = get_jwt_identity()
    
    try:
        # Check if user has submitted rankings
        rankings = list(mongo.db.rankings.find({'user_id': user_id}))
        
        # Get query parameters
        top_n = request.args.get('limit', 10, type=int)
        skip_ranking_check = request.args.get('skip_ranking_check', 'false').lower() == 'true'
        
        if not rankings and not skip_ranking_check:
            # Return a friendly message, but also allow the frontend to ignore this
            # by passing ?skip_ranking_check=true in the query string
            return jsonify({
                "error": "No rankings found. Please rank perfumes first.",
                "message": "You can still get recommendations based on our pre-trained models by adding ?skip_ranking_check=true to your request."
            }), 400
            
        # Generate recommendations using the ML models
        perfume_ids = generate_recommendations(user_id, top_n=top_n)
        
        # Fetch perfume details for recommended perfumes
        recommended_perfumes = []
        for perfume_id in perfume_ids:
            perfume = mongo.db.perfumes.find_one({'_id': perfume_id})
            if perfume:
                recommended_perfumes.append(perfume)
                
        # Format response
        formatted_perfumes = json.loads(dumps(recommended_perfumes))
        
        return jsonify(formatted_perfumes), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500 