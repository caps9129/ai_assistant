import os
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
# Load keys from the environment, defaulting to None if not found
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
Maps_API_KEY = os.getenv("Maps_API_KEY")

if not OPENAI_API_KEY or not Maps_API_KEY:
    print("⚠️ WARNING: One or more API keys are missing from your .env file.")

# --- Project Root Path ---
# This line is key: It finds the directory where config.py is located,
# which we define as the project's root.
PROJECT_ROOT = Path(__file__).parent

# --- Credential Paths ---
CREDENTIALS_DIR = PROJECT_ROOT / "credentials"
GOOGLE_CREDENTIALS_PATH = CREDENTIALS_DIR / "google_credentials.json"
GOOGLE_TOKEN_PATH = CREDENTIALS_DIR / \
    "google_token.json"  # Where the token will be saved

# --- Model Paths ---
MODELS_DIR = PROJECT_ROOT / "models"
# (You can add paths to your wakeword, punctuation, and other models here too)
# WAKEWORD_MODEL_PATH = MODELS_DIR / "wakeword" / "hey_jarvis.onnx"

MEMORY_LOG_DIR = PROJECT_ROOT / "memory_logs"


class Language(Enum):
    ENGLISH = "en-US"
    CHINESE = "zh-TW"
    FRENCH = "fr-FR"
    SPANISH_SPAIN = "es-ES"
    SPANISH_LATAM = "es-US"
    KOREAN = "ko-KR"
    JAPANESE = "ja-JP"
