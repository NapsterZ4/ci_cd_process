from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import status
from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.utilities.s3_storage import S3ModelStorage
from api.services.model_service import ModelService
from api.schemas.prediction import UserQuestion
from api.services.gpt_service import OpenAIClient, EmbeddingService
from api.utilities.load_files import load_prompt
from api.services.retriever import Retriever


router = APIRouter(prefix="/predict", tags=["Prediccion"])
retriever = Retriever(docs_dir="context/")
embedding_service = EmbeddingService()

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

@router.post("/gpt")
def gpt(request: UserQuestion) -> JSONResponse:
    client = OpenAIClient()
    context = retriever.search(request.question, k=5)

    svc = EmbeddingService()
    embed_query = svc.embed_query(request.question)
    embed_context = svc.embed_query(context)

    similarity = svc.cosine_similarity(embed_query, embed_context)

    if similarity < 0.25:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "message": "No se encontraron resultados para tu pregunta.",
                "similarity": similarity
            }
        )

    prompt = f"Contexto:\n{context}\n\nPregunta: {request.question}"
    resp = client.ask(
        prompt=prompt,
        system=load_prompt("prompts/prompt_1.md")
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": request.question,
            "response": resp,
            "query_embedding": embed_query,
            "context_embedding": embed_context,
            "similarity": similarity
        }
    )
