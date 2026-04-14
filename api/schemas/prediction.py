from pydantic import BaseModel, Field
from typing import TypedDict


class PredictionRequest(BaseModel):
    edad: int = Field(..., examples=[71])
    tamano_tumor_mm: float = Field(..., examples=[12.1])
    forma_tumor: str = Field(..., examples=["Lobular"])
    textura: str = Field(..., examples=["Lisa"])
    marcador_ca125: float = Field(..., examples=[36.0])
    marcador_cea: float = Field(..., examples=[1.96])
    densidad_celular: float = Field(..., examples=[45.1])
    historial_familiar: int = Field(..., examples=[0])
    fumador: int = Field(..., examples=[0])


class UserQuestion(BaseModel):
    question: str

class PredictionResponse(BaseModel):
    prediccion: str
    clase: int

class Consulta(BaseModel):
    mensaje: str
    session_id: str = "default"

class Respuesta(BaseModel):
    mensaje: str
    agente: str

class Estado(TypedDict):
    messages: list
    siguiente: str
