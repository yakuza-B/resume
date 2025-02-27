import os
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["B"]  # Your database name
collection = db["agriculture"]  # Your collection name

print("MongoDB Connected Successfully!")
