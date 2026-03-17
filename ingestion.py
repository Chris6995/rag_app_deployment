import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import File, UploadFile
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

# cargar las variables de entorno desde un archivo .env para configurar la conexión a Qdrant y el modelo de embedding
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents").strip()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()

# función para obtener un cliente de Qdrant, utilizando las credenciales y URL configurados en las variables de entorno
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

# configurar el modelo de embedding para que esté disponible globalmente en la aplicación, evitando la necesidad de inicializarlo repetidamente
def configure_embed_model():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

# construir un doc_id único basado en el contenido del archivo y su nombre, para evitar duplicados y facilitar la gestión de documentos
def build_doc_id(file_path: Path, filename: str):
    digest = hashlib.md5(file_path.read_bytes()).hexdigest()[:12]
    safe_name = Path(filename).stem.replace(" ", "_").lower()
    return f"{safe_name}-{digest}"

# crear colleción a Qdrant si no existe, con la configuración adecuada para los vectores de embedding
def create_collection():
    client = get_qdrant_client() # obtener un cliente de Qdrant para interactuar con la base de datos de vectores
    vector_size = len(Settings.embed_model.get_text_embedding("test")) # obtener el tamaño del vector de embedding para configurar la colección correctamente

    try: # intentar crear la colección en Qdrant, si ya existe se ignora el error
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
    except Exception as exc:
        if "exists" not in str(exc).lower():
            raise

    try: # crear un índice de payload en Qdrant para el campo "doc_id", lo que permite realizar búsquedas eficientes basadas en este campo
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="rag_doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception as exc:
        if "exists" not in str(exc).lower():
            raise

# función principal para ingerir un archivo, que lee el contenido, genera embeddings, y los almacena en Qdrant con la metadata adecuada
def ingest_file(file_path: Path, filename: str, chunk_size: int, chunk_overlap: int):
    configure_embed_model()
    create_collection()
    Settings.text_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # leer el contenido del archivo utilizando SimpleDirectoryReader, lo que permite manejar diferentes formatos de documentos de manera sencilla
    documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
    doc_id = build_doc_id(file_path, filename)

    for doc in documents:
        doc.metadata["rag_doc_id"] = doc_id
        doc.metadata["source_file"] = filename
    # obtener un cliente de Qdrant y configurar el vector store para almacenar los embeddings generados a partir del contenido del documento, utilizando la colección configurada previamente
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # crear un índice de vectores a partir de los documentos leídos, lo que implica generar embeddings para cada documento y almacenarlos en Qdrant junto con su metadata
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_indexed": len(index.docstore.docs),
    }

# endpoint de FastAPI para manejar la carga de archivos, que recibe un archivo, lo guarda temporalmente, y luego llama a la función de ingestión para procesarlo y almacenarlo en Qdrant
async def ingest_document_file(
    file: UploadFile = File(...),
    chunk_size: int = 700,
    chunk_overlap: int = 80,
):
    suffix = Path(file.filename).suffix or ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        return ingest_file(temp_path, file.filename, chunk_size, chunk_overlap)
    finally:
        temp_path.unlink(missing_ok=True)
