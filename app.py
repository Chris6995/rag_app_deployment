import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="centered")
st.title("Chatbot RAG con LlamaIndex")
st.caption("Sube un documento, indexa su contenido en Qdrant y pregúntale sobre él.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None


def ingest_file(uploaded_file):
    response = requests.post(
        f"{API_URL}/ingest",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def ask_question(question: str, doc_id: str):
    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question, "doc_id": doc_id},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.subheader("Documento")
    uploaded_file = st.file_uploader(
        "Sube un archivo",
        type=["pdf", "txt", "md", "docx"],
    )

    if st.button("Indexar documento", use_container_width=True):
        if uploaded_file is None:
            st.warning("Primero sube un documento.")
        else:
            with st.spinner("Indexando documento..."):
                try:
                    result = ingest_file(uploaded_file)
                    st.session_state.doc_id = result["doc_id"]
                    st.session_state.messages = []
                    st.success(
                        f"Documento indexado. doc_id={result['doc_id']} | chunks={result['chunks_indexed']}"
                    )
                except requests.HTTPError as exc:
                    detail = exc.response.json().get("detail", str(exc))
                    st.error(detail)
                except Exception as exc:
                    st.error(str(exc))

    if st.session_state.doc_id:
        st.info(f"Documento activo: {st.session_state.doc_id}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Haz una pregunta sobre el documento")

if prompt:
    if not st.session_state.doc_id:
        st.warning("Antes de chatear tienes que subir e indexar un documento.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generando respuesta..."):
                try:
                    result = ask_question(prompt, st.session_state.doc_id)
                    answer = result["answer"]
                except requests.HTTPError as exc:
                    answer = exc.response.json().get("detail", str(exc))
                except Exception as exc:
                    answer = str(exc)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
