# Sistema RAG con LlamaIndex, FastAPI y Streamlit

Proyecto simple de RAG con:

- `LlamaIndex` para la lógica RAG
- `Qdrant Cloud` como base de datos vectorial
- `Groq`, `Gemini` u `OpenRouter` como proveedor LLM configurable
- `FastAPI` como backend
- `Streamlit` como frontend tipo chatbot

## Estructura

```bash
.
├── app.py
├── backend.py
├── ingestion.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Qué hace

1. Subes un documento desde Streamlit.
2. El frontend lo envía al backend FastAPI.
3. `ingestion.py` extrae el texto, lo trocea con LlamaIndex y lo guarda en Qdrant Cloud.
4. El chat consulta solo sobre el documento activo usando `doc_id`.

## Configuración

### 1. Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Crear variables de entorno

```bash
cp .env.example .env
```

Configura como mínimo:

```bash
QDRANT_URL=...
QDRANT_API_KEY=...
LLM_PROVIDER=groq
GROQ_API_KEY=...
LLM_MODEL=llama-3.3-70b-versatile
```

### 4. Proveedores soportados

#### Groq

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=tu_api_key
LLM_MODEL=llama-3.3-70b-versatile
```

#### Gemini

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=tu_api_key
LLM_MODEL=models/gemini-1.5-flash
```

#### OpenRouter

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=tu_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

## Ejecutar el proyecto

### 1. Arrancar FastAPI

```bash
uvicorn backend:app --reload
```

API disponible en:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

### 2. Arrancar Streamlit

```bash
streamlit run app.py
```

## Endpoints

### `POST /ingest`

Recibe un archivo y devuelve:

```json
{
  "doc_id": "mi_archivo-123abc456def",
  "filename": "mi_archivo.pdf",
  "chunks_indexed": 12
}
```

### `POST /chat`

Body:

```json
{
  "question": "Resume el documento",
  "doc_id": "mi_archivo-123abc456def"
}
```

## Notas

- La colección de Qdrant se crea automáticamente en la primera ingesta.
- Se usa un único collection en Qdrant y se filtra por `doc_id`.
- Los embeddings son locales con `sentence-transformers`, así no dependes del proveedor LLM para vectorizar.
- `backend.py` se centra en retrieval y generación de respuesta.
- `ingestion.py` se centra solo en la ingesta del documento.
- Si quieres reiniciar los datos indexados, puedes vaciar la colección en Qdrant Cloud.
