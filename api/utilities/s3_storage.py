import json
import logging
import tempfile
import boto3
from pathlib import Path
import joblib
from .interfaces import IStorage

logger = logging.getLogger(__name__)


class S3ModelStorage(IStorage):

    def __init__(self, bucket: str, prefix: str = "cancer-model"):
        self._s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix

    def _build_key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def download_dataset(
            self,
            model_name: str,
            local_path: str = "data/dataset_cancer.csv"
    ) -> str:
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(
            Bucket=self.bucket,
            Key=model_name,
            Filename=str(local)
        )
        logger.info(f"Dataset descargado: s3://{self.bucket}/{model_name} -> {local}")
        return str(local)

    def save_model(
            self,
            model,
            metrics: dict,
            model_name: str = "modelo_cancer_v1.joblib",
            metrics_name: str = "metrics.json"
    ) -> None:
        model_key = self._build_key(model_name)
        metrics_key = self._build_key(metrics_name)

        with tempfile.NamedTemporaryFile(
                suffix=".joblib",
                delete=False
        ) as tmp:
            joblib.dump(model, tmp.name)
            self._s3.upload_file(
                Filename=tmp.name,
                Bucket=self.bucket,
                Key=model_key
            )

        with tempfile.NamedTemporaryFile(
                suffix=".json",
                mode="w",
                delete=False
        ) as tmp:
            json.dump(metrics, tmp, indent=2)
            tmp.flush()
            self._s3.upload_file(
                Filename=tmp.name,
                Bucket=self.bucket,
                Key=metrics_key
            )

        logger.info(f"Modelo subido: s3://{self.bucket}/{model_key}")

    def load_model(self, model_name: str = "modelo_cancer_v1.joblib"):
        model_key = self._build_key(model_name)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            self._s3.download_file(
                Bucket=self.bucket,
                Key=model_key,
                Filename=tmp.name
            )
            return joblib.load(tmp.name)
