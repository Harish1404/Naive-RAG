from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # lowercase aliases — this is what ChatService / ChainService actually read
    gemini_api_key = GEMINI_API_KEY
    groq_api_key = GROQ_API_KEY

settings = Settings()

