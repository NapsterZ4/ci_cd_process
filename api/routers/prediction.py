from sys import prefix

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import status
from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.utilities.s3_storage import S3ModelStorage
from api.services.model_service import ModelService

router = APIRouter(prefix="/predict", tags=["Prediccion"])


s3_client = S3ModelStorage(
    bucket="general-mlops-versioning",
    prefix="/"
)
model_service = ModelService(
    s3_client=s3_client,
    model_name="modelo_cancer_v1.joblib",
    local_path="/tmp/modelo_cancer_v1.joblib"
)

@router.post("/", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    clase, prediccion = model_service.predict(request.model_dump())
    return PredictionResponse(prediccion=prediccion, clase=clase)


@router.get("/health", tags=["Health"])
def health() -> JSONResponse:
    return JSONResponse(
        content={"status": "ok"},
        status_code=status.HTTP_200_OK
    )
