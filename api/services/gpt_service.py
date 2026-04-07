from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from config import OPENAI_API_KEY
import numpy as np


class OpenAIClient:
    def __init__(
            self,
            model: str = "gpt-4o-mini",
            temperature: float = 0.0
    ):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=OPENAI_API_KEY
        )

    def ask(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        return self.llm.invoke(messages).content


class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def embed_documents(self, paths: list[str]) -> list[list[float]]:
        docs = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read().strip())
        return self.embeddings.embed_documents(docs)

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
