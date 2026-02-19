from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UPLOAD_FOLDER = './temp'

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Rate Limiting
RATELIMIT_ENABLED = os.getenv("RATELIMIT_ENABLED", "true").lower() == "true"
RATELIMIT_STORAGE_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"
