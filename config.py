import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "model/modelo_cancer_v1.joblib")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
