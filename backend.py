import os
import sys
import logging
import time

# --- CONFIGURACIÓN DE LOGS (Haz esto lo primero) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)] # Obliga a enviar a la consola de Render
)
logger = logging.getLogger("RAG_BACKEND")
logger.info("Iniciando proceso de carga de módulos...")


from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import set_global_handler
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler

logger.info("✅ Módulos de LlamaIndex cargados correctamente")

load_dotenv()

from langfuse import Langfuse

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

langfuse = None
langfuse_handler = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
    langfuse_handler = LlamaIndexCallbackHandler(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
    Settings.callback_manager = CallbackManager([langfuse_handler])
    logger.info("✅ Langfuse configurado")
else:
    logger.warning("⚠️ Langfuse no configurado: faltan claves")

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents").strip()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip()

SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "4"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))

rerank_postprocessor = CohereRerank(
    api_key=COHERE_API_KEY,
    model="rerank-multilingual-v3.0",
    top_n=RERANK_TOP_N
    # input_type="search_query", # Importante para RAG
)
LLM_TEMPERATURE = 0

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
    from qdrant_client import QdrantClient
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
    from llama_index.llms.openai_like import OpenAILike
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY.")

 
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=COHERE_API_KEY,
        model_name="embed-multilingual-v3.0",
        input_type="search_query", # Importante para RAG
    )
    logger.info("✅ Modelo de Embedding listo")
    Settings.llm = OpenAILike(
        model=LLM_MODEL,
        api_base="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
        is_chat_model=True,
    )
    logger.info("✅ LLM configurado")


def get_query_engine(doc_id: str):
    # from llama_index.core.postprocessor import SentenceTransformerRerank
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    configure_models()

    logger.info("✅ Reranker listo")

    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
    )

    logger.info(f"Creando índice para doc_id: {doc_id}...")

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
        node_postprocessors=[rerank_postprocessor],
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": PromptTemplate(QA_TEMPLATE)}
    )
    logger.info("✅ Query Engine generado con éxito")
    return query_engine
    # chat_engine = index.as_chat_engine(
    #     chat_mode="context",
    #     memory=memory,
    #     llm=Settings.llm,
    #     node_postprocessors=[rerank_postprocessor],
    #     system_prompt=(
    #         "Eres un asistente técnico. Responde de forma concisa usando el contexto "
    #         "proporcionado y el historial de la conversación."
    #     ),
    #     # Filtramos por el documento específico como hacías antes
    #     filters=MetadataFilters(filters=[MetadataFilter(key="rag_doc_id", value=doc_id)])
    # )

    # return chat_engine


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_route(
    file: UploadFile = File(...),
    chunk_size: int = Form(700),
    chunk_overlap: int = Form(80),
):
    try:
        from ingestion import ingest_document_file
        return await ingest_document_file(file, chunk_size, chunk_overlap)
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
