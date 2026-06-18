import streamlit as st

from src.document_loader import (
    is_valid_file,
    save_uploaded_file,
    extract_text
)

from src.ollama_client import OllamaClient
from src.text_chunker import chunk_text

from src.embeddings import EmbeddingModel

from src.vector_store import VectorStore

from src.retriever import Retriever


# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📄",
    layout="wide"
)


# ----------------------------------
# LOAD EMBEDDING MODEL
# ----------------------------------

@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()


embedding_model = load_embedding_model()
llm_client = OllamaClient()


# ----------------------------------
# UI
# ----------------------------------

st.title("📄 RAG Document QA System")

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True
)


# ----------------------------------
# PROCESS DOCUMENTS
# ----------------------------------

if uploaded_files:

    all_chunks = []

    for file in uploaded_files:

        if not is_valid_file(file.name):
            st.error(f"Invalid file: {file.name}")
            continue

        file_path = save_uploaded_file(file)

        st.success(f"Saved: {file.name}")

        try:

            # --------------------------
            # TEXT EXTRACTION
            # --------------------------

            extracted_text = extract_text(file_path)

            st.subheader(
                f"Extracted Text - {file.name}"
            )

            st.text_area(
                "Extracted Content",
                extracted_text[:2000],
                height=250,
                label_visibility="collapsed"
            )

            # --------------------------
            # CHUNKING
            # --------------------------

            chunks = chunk_text(
                extracted_text,
                chunk_size=500,
                overlap=100
            )

            all_chunks.extend(chunks)

            st.subheader(
                f"Chunk Information - {file.name}"
            )

            st.write(
                f"Total Chunks: {len(chunks)}"
            )

            for i, chunk in enumerate(chunks):

                with st.expander(
                    f"{file.name} - Chunk {i+1}"
                ):
                    st.write(chunk)

        except Exception as e:

            st.error(
                f"Error processing {file.name}: {str(e)}"
            )

    # ----------------------------------
    # EMBEDDINGS
    # ----------------------------------

    if len(all_chunks) > 0:

        chunk_embeddings = (
            embedding_model.encode_chunks(
                all_chunks
            )
        )

        st.subheader(
            "Embedding Information"
        )

        st.write(
            f"Number of Embeddings: {len(chunk_embeddings)}"
        )

        st.write(
            f"Embedding Dimension: {len(chunk_embeddings[0])}"
        )

        st.subheader(
            "First Embedding Preview"
        )

        st.write(
            chunk_embeddings[0][:10]
        )

        # ----------------------------------
        # FAISS
        # ----------------------------------

        vector_store = VectorStore()

        vector_store.create_index(
            chunk_embeddings,
            all_chunks
        )

        vector_store.save()

        st.subheader(
            "FAISS Information"
        )

        st.write(
            f"Vectors Stored: {len(all_chunks)}"
        )

        st.success(
            "FAISS Index Created Successfully"
        )

        # ----------------------------------
        # RETRIEVER
        # ----------------------------------

        retriever = Retriever(
            vector_store,
            embedding_model
        )

        st.divider()

        st.header(
            "Ask Questions"
        )


        question = st.text_input(
            "Ask a question about the uploaded documents"
        )

        if question:

            retrieved_chunks = (
                retriever.retrieve(
                    question,
                    top_k=3
                )
            )

            context = "\n\n".join(
                retrieved_chunks
            )

            answer = (
                llm_client.generate_answer(
                    question,
                    context
                )
            )

            st.subheader("Answer")

            st.write(answer)

            with st.expander(
                "View Retrieved Sources"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks
                ):

                    st.markdown(
                        f"### Source {i+1}"
                    )

                    st.write(chunk)
    