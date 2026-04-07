import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Retriever:
    def __init__(self, docs_dir: str, model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model)
        self.vectorstore = self._build_index(docs_dir)

    def _build_index(self, docs_dir: str) -> FAISS:
        texts = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        for filename in os.listdir(docs_dir):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = splitter.split_text(f.read())
                texts.extend(chunks)
        return FAISS.from_texts(texts, self.embeddings)

    def search(self, query: str, k: int = 3) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)
