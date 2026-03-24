from abc import ABC, abstractmethod


class IStorage(ABC):
    @abstractmethod
    def download_dataset(
            self,
            model_name: str,
            local_path: str = "data/dataset_cancer.csv"
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_model(
            self,
            model,
            metrics: dict,
            model_name: str | None = None,
            metrics_key: str | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_key: str | None = None):
        raise NotImplementedError
