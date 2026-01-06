import os
import redis
from qdrant_client import QdrantClient
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ---------- QDRANT ----------
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# ---------- REDIS ----------
redis_client = redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=True
)

# ---------- MONGODB ----------
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["rag_database"]
bookings_collection = db["bookings"]

print("✅ Databases connected successfully")
