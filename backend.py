import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ingestion import ingest_document_file


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "4"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

QA_TEMPLATE = """Context information is below:
---------------------
{context_str}
---------------------
Given the context information above I want you to think
step by step to answer the query in a crisp manner,
incase you don't know the answer say 'I don't know!'

Query: {query_str}

Answer:"""


app = FastAPI(title="Simple RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def get_qdrant_client():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY.")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    def search(*, collection_name, query_vector, limit=10, query_filter=None, with_payload=True, with_vectors=False, **kwargs):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            **kwargs,
        )
        return response.points or []

    client.search = search

    return client


def configure_models():
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY.")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = OpenAILike(
        model=LLM_MODEL,
        api_base="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
        is_chat_model=True,
    )


def get_query_engine(doc_id: str):
    configure_models()

    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

    filters = MetadataFilters(
        filters=[MetadataFilter(key="rag_doc_id", value=doc_id)]
    )

    query_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        filters=filters,
        llm=Settings.llm,
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": PromptTemplate(QA_TEMPLATE)}
    )

    return query_engine


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_route(file: UploadFile = File(...)):
    try:
        return await ingest_document_file(file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat")
async def chat_with_document(payload: dict):
    try:
        question = payload.get("question", "")
        doc_id = payload.get("doc_id", "")

        if not question or not doc_id:
            raise ValueError("You must send 'question' and 'doc_id'.")

        query_engine = get_query_engine(doc_id)
        response = query_engine.query(question)

        if not getattr(response, "source_nodes", None):
            raise ValueError(
                "No se encontraron fragmentos relevantes para ese documento. "
                "Asegura que el archivo se ha indexado correctamente."
            )

        return {"answer": str(response), "doc_id": doc_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
