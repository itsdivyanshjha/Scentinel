// MongoDB initialization script
db = db.getSiblingDB('scentinel');

// Create collections
db.createCollection('users');
db.createCollection('perfumes');
db.createCollection('rankings');

// Create indexes
db.users.createIndex({ "email": 1 }, { unique: true });
db.rankings.createIndex({ "user_id": 1, "perfume_id": 1 }, { unique: true });

// Log completion
print('MongoDB initialized for Scentinel'); 