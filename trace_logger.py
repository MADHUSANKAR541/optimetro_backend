from pymongo import MongoClient
from datetime import datetime
import os

MONGO_URI = os.environ.get("MONGODB_ATLAS_URI", "your-mongodb-atlas-uri")
DB_NAME = "optimetro"
COLLECTION_NAME = "explainability_traces"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def log_trace(trace: dict):
    try:
        trace["timestamp"] = datetime.utcnow().isoformat() + "Z"
        collection.insert_one(trace)
    except Exception as e:
        print(f"Failed to log trace to MongoDB: {e}")
