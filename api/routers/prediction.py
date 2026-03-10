from fastapi import APIRouter

from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.services.model_service import model_service

router = APIRouter(prefix="/predict", tags=["Prediccion"])


@router.post("/", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    clase, prediccion = model_service.predict(request.model_dump())
    return PredictionResponse(prediccion=prediccion, clase=clase)
