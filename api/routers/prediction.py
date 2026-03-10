from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import status
from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.services.model_service import model_service

router = APIRouter(prefix="/predict", tags=["Prediccion"])


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
