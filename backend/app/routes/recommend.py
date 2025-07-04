from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import mongo
from app.services.recommendation_service import generate_recommendations, get_cold_start_recommendations
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
        enable_diversity = request.args.get('enable_diversity', 'true').lower() == 'true'
        diversity_weight = request.args.get('diversity_weight', 0.3, type=float)
        
        if not rankings and not skip_ranking_check:
            # Return a friendly message, but also allow the frontend to ignore this
            # by passing ?skip_ranking_check=true in the query string
            return jsonify({
                "error": "No rankings found. Please rank perfumes first.",
                "message": "You can still get recommendations based on our pre-trained models by adding ?skip_ranking_check=true to your request."
            }), 400
            
        # Generate recommendations using the ML models
        perfume_ids = generate_recommendations(user_id, top_n=top_n, enable_diversity=enable_diversity, diversity_weight=diversity_weight)
        
        # Fetch perfume details for recommended perfumes
        recommended_perfumes = []
        for perfume_id in perfume_ids:
            perfume = mongo.db.perfumes.find_one({'_id': perfume_id})
            if perfume:
                # Transform field names to match frontend expectations
                transformed = {
                    '_id': str(perfume['_id']),
                    'name': perfume.get('Name', perfume.get('name', '')),
                    'brand': perfume.get('Brand', perfume.get('brand', '')),
                    'notes': perfume.get('Notes', perfume.get('notes', '')),
                    'description': perfume.get('Description', perfume.get('description', '')),
                    'image_url': perfume.get('Image URL', perfume.get('image_url', '')),
                    'gender': perfume.get('Gender', perfume.get('gender', ''))
                }
                recommended_perfumes.append(transformed)
                
        # Format response
        return jsonify(recommended_perfumes), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@recommend_bp.route('/cold-start', methods=['GET'])
def cold_start_recommendations():
    """Get recommendations for new users (no authentication required)"""
    try:
        # Get query parameters
        top_n = request.args.get('limit', 10, type=int)
        enable_diversity = request.args.get('enable_diversity', 'true').lower() == 'true'
        diversity_weight = request.args.get('diversity_weight', 0.3, type=float)
        
        # Generate recommendations using pre-trained models
        perfume_ids = get_cold_start_recommendations(top_n=top_n, enable_diversity=enable_diversity, diversity_weight=diversity_weight)
        
        # Fetch perfume details for recommended perfumes
        recommended_perfumes = []
        for perfume_id in perfume_ids:
            perfume = mongo.db.perfumes.find_one({'_id': perfume_id})
            if perfume:
                # Transform field names to match frontend expectations
                transformed = {
                    '_id': str(perfume['_id']),
                    'name': perfume.get('Name', perfume.get('name', '')),
                    'brand': perfume.get('Brand', perfume.get('brand', '')),
                    'notes': perfume.get('Notes', perfume.get('notes', '')),
                    'description': perfume.get('Description', perfume.get('description', '')),
                    'image_url': perfume.get('Image URL', perfume.get('image_url', '')),
                    'gender': perfume.get('Gender', perfume.get('gender', ''))
                }
                recommended_perfumes.append(transformed)
                
        # Format response
        return jsonify(recommended_perfumes), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@recommend_bp.route('/popular', methods=['GET'])
def get_popular_perfumes():
    """Get most popular perfumes based on rankings"""
    try:
        # Get query parameters
        top_n = request.args.get('limit', 10, type=int)
        
        # Aggregate rankings to find most popular perfumes
        pipeline = [
            {"$group": {"_id": "$perfume_id", "avg_rank": {"$avg": "$rank"}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "avg_rank": 1}},
            {"$limit": top_n}
        ]
        
        popular_perfumes = list(mongo.db.rankings.aggregate(pipeline))
        
        # Fetch full perfume details
        recommended_perfumes = []
        for item in popular_perfumes:
            perfume = mongo.db.perfumes.find_one({'_id': item['_id']})
            if perfume:
                # Transform field names to match frontend expectations
                transformed = {
                    '_id': str(perfume['_id']),
                    'name': perfume.get('Name', perfume.get('name', '')),
                    'brand': perfume.get('Brand', perfume.get('brand', '')),
                    'notes': perfume.get('Notes', perfume.get('notes', '')),
                    'description': perfume.get('Description', perfume.get('description', '')),
                    'image_url': perfume.get('Image URL', perfume.get('image_url', '')),
                    'gender': perfume.get('Gender', perfume.get('gender', '')),
                    'popularity': {
                        'ranking_count': item['count'],
                        'average_rank': item['avg_rank']
                    }
                }
                recommended_perfumes.append(transformed)
                
        # Format response
        return jsonify(recommended_perfumes), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500 