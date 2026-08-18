from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    FLUX_AI = os.getenv("FLUX_AI")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

    GITHUB_PAT = os.getenv("GITHUB_PAT")

    MONGO_URL = os.getenv("MONGO_URL")
    DB_NAME = os.getenv("DB_NAME")

    # lowercase aliases — this is what ChatService / ChainService actually read
    gemini_api_key = GEMINI_API_KEY
    groq_api_key = GROQ_API_KEY
    flux_ai = FLUX_AI

settings = Settings()

