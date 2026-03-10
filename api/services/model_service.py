import joblib
import pandas as pd
from config import MODEL_PATH

LABEL_MAP = {1: "Maligno", 0: "Benigno"}


class ModelService:
    def __init__(self) -> None:
        self._model = joblib.load(MODEL_PATH)

    def predict(self, data: dict) -> tuple[int, str]:
        df = pd.DataFrame([data])
        clase = int(self._model.predict(df)[0])
        return clase, LABEL_MAP[clase]


model_service = ModelService()
