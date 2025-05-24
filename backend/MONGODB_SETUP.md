# MongoDB Setup for Scentinel

This document provides instructions for setting up and configuring MongoDB Atlas for the Scentinel perfume recommendation system.

## Connection Status

There are currently authentication issues with the MongoDB Atlas connection. Follow the troubleshooting steps below to resolve them.

## Configuration Files Updated

The following files have been updated to use the MongoDB Atlas connection:

1. `backend/app/__init__.py` - Updated to use the MONGODB_URI environment variable
2. `backend/app/utils/init_db.py` - Updated to use the MONGODB_URI for database initialization
3. `backend/test_mongodb_connection.py` - Created to test the MongoDB connection

## Required Collections

The system requires the following MongoDB collections:

1. `users` - Stores user profiles, credentials, and preferences
2. `perfumes` - Stores the perfume dataset with attributes
3. `recommendations` - Stores generated recommendations
4. `rankings` - Stores user rankings and interactions

## Troubleshooting Steps

If you're encountering authentication issues:

1. **Verify Credentials**:
   - Log into MongoDB Atlas dashboard
   - Go to "Database Access" section
   - Verify the "jhadivyansh29" user exists with proper permissions
   - Reset the password if needed

2. **Check Network Access**:
   - Go to "Network Access" in MongoDB Atlas
   - Ensure your current IP address is in the whitelist
   - Consider adding "0.0.0.0/0" for development (not recommended for production)

3. **Validate Connection String**:
   - Make sure the connection string format is correct
   - Check that the cluster name, username, and password are accurate
   - Verify the database name is included in the URI

4. **Test Connection**:
   - Run `python backend/test_mongodb_connection.py` to verify the connection

## Environment Setup

Create a `.env` file in the `backend` directory with the following content:

```
MONGODB_URI=mongodb+srv://<username>:<password>@scentinelcluster.apxy5nv.mongodb.net/scentinel?retryWrites=true&w=majority&appName=ScentinelCluster
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
```

Replace `<username>` and `<password>` with your actual MongoDB Atlas credentials.

## Next Steps

After resolving the authentication issues:

1. Run the database initialization script to load perfume data:
   ```
   python backend/init_db.py
   ```

2. Start the Flask backend:
   ```
   cd backend
   python run.py
   ```

3. Run the pre-training script to initialize recommendation models:
   ```
   python standalone_pretrain.py
   ``` 