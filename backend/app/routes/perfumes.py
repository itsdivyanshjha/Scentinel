from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import mongo
import random
from bson.json_util import dumps
import json

perfumes_bp = Blueprint('perfumes', __name__, url_prefix='/api/perfumes')

@perfumes_bp.route('/list', methods=['GET'])
@jwt_required()
def get_perfumes_to_rank():
    """Return a list of 10 random perfumes for the user to rank"""
    try:
        # Get a random sample of 10 perfumes from the database
        pipeline = [{"$sample": {"size": 10}}]
        perfumes = list(mongo.db.perfumes.aggregate(pipeline))
        
        # Format the response
        formatted_perfumes = json.loads(dumps(perfumes))
        
        return jsonify(formatted_perfumes), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@perfumes_bp.route('/rank', methods=['POST'])
@jwt_required()
def submit_rankings():
    """Submit user rankings for perfumes"""
    data = request.get_json()
    user_id = get_jwt_identity()
    
    # Validate input
    if not data or 'rankings' not in data:
        return jsonify({"error": "Rankings are required"}), 400
    
    rankings = data['rankings']
    
    # Validate rankings format
    if not isinstance(rankings, list):
        return jsonify({"error": "Rankings must be a list"}), 400
        
    # Check if each ranking has perfume_id and rank
    for ranking in rankings:
        if not isinstance(ranking, dict) or 'perfume_id' not in ranking or 'rank' not in ranking:
            return jsonify({"error": "Each ranking must have perfume_id and rank"}), 400
    
    # Store rankings in the database
    for ranking in rankings:
        mongo.db.rankings.update_one(
            {'user_id': user_id, 'perfume_id': ranking['perfume_id']},
            {'$set': {'rank': ranking['rank']}},
            upsert=True
        )
    
    return jsonify({"message": "Rankings submitted successfully"}), 200

@perfumes_bp.route('/all', methods=['GET'])
def get_all_perfumes():
    """Return all perfumes in the database (paginated)"""
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        skip = (page - 1) * limit
        
        # Get perfumes with pagination
        perfumes = list(mongo.db.perfumes.find().skip(skip).limit(limit))
        total = mongo.db.perfumes.count_documents({})
        
        # Format the response
        formatted_perfumes = json.loads(dumps(perfumes))
        
        return jsonify({
            'perfumes': formatted_perfumes,
            'total': total,
            'page': page,
            'limit': limit,
            'pages': (total + limit - 1) // limit
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500 