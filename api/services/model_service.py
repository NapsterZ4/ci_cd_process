import joblib
import pandas as pd
from api.utilities.interfaces import IStorage

LABEL_MAP = {1: "Maligno", 0: "Benigno"}


class ModelService:

    def __init__(
            self,
            s3_client: IStorage,
            model_name: str,
            local_path: str = "/tmp/modelo_cancer_v1.joblib"
    ) -> None:
        self._local_path = local_path
        s3_client.download_dataset(model_name=model_name, local_path=local_path)
        self._model = joblib.load(local_path)

    def predict(self, data: dict) -> tuple[int, str]:
        df = pd.DataFrame([data])
        prediction_class = int(self._model.predict(df)[0])
        return prediction_class, LABEL_MAP[prediction_class]
