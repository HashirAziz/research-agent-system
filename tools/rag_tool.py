"""
tools/rag_tool.py
─────────────────
ChromaDB-backed RAG tool for private document retrieval.

Supports:
- Ingesting PDFs, DOCX, TXT files
- Persistent vector store
- Similarity + MMR retrieval
- Metadata-aware results
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import settings
from schemas.outputs import Source
from utils.logger import get_logger

logger = get_logger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    """Return OpenAI embeddings model."""
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model="text-embedding-3-small",
    )


def get_vector_store() -> Chroma:
    """Return (or create) the persistent ChromaDB vector store."""
    persist_dir = settings.chroma_persist_dir
    os.makedirs(persist_dir, exist_ok=True)

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )


def load_document(file_path: str) -> list[Document]:
    """Load a single document based on file extension."""
    path = Path(file_path)
    ext = path.suffix.lower()

    loaders = {
        ".pdf":  PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt":  TextLoader,
        ".md":   TextLoader,
    }

    loader_cls = loaders.get(ext)
    if not loader_cls:
        logger.warning(f"Unsupported file type: {ext}. Skipping {file_path}")
        return []

    try:
        loader = loader_cls(file_path)
        docs = loader.load()
        logger.info(f"📄 Loaded {len(docs)} pages from '{path.name}'")
        return docs
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []


def ingest_documents(file_paths: list[str]) -> int:
    """
    Ingest documents into ChromaDB vector store.
    
    Returns:
        Number of chunks ingested.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs: list[Document] = []
    for fp in file_paths:
        docs = load_document(fp)
        chunks = splitter.split_documents(docs)
        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source_file"] = Path(fp).name
        all_docs.extend(chunks)

    if not all_docs:
        logger.warning("No documents to ingest.")
        return 0

    store = get_vector_store()
    store.add_documents(all_docs)

    logger.info(f"✅ Ingested {len(all_docs)} chunks into ChromaDB")
    return len(all_docs)


def ingest_directory(directory: str = "./data/documents") -> int:
    """Ingest all supported documents from a directory."""
    supported = [".pdf", ".docx", ".txt", ".md"]
    files = [
        str(p) for p in Path(directory).rglob("*")
        if p.suffix.lower() in supported and p.is_file()
    ]
    if not files:
        logger.info(f"No documents found in {directory}")
        return 0
    return ingest_documents(files)


def rag_search(
    query: str,
    k: int = 5,
    use_mmr: bool = True,
) -> list[Document]:
    """
    Retrieve relevant documents from ChromaDB.
    
    Args:
        query: Search query
        k: Number of results
        use_mmr: Use Maximal Marginal Relevance for diversity
    """
    store = get_vector_store()
    
    # Check if collection has documents
    try:
        collection = store._collection
        count = collection.count()
        if count == 0:
            logger.info("RAG: Vector store is empty — skipping RAG retrieval")
            return []
    except Exception:
        return []

    try:
        if use_mmr:
            docs = store.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
        else:
            docs = store.similarity_search(query, k=k)
        
        logger.info(f"📚 RAG: '{query[:60]}' → {len(docs)} chunks retrieved")
        return docs
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return []


def docs_to_sources(docs: list[Document]) -> list[Source]:
    """Convert LangChain Documents to Source objects."""
    sources = []
    for doc in docs:
        title = doc.metadata.get("source_file", "Private Document")
        sources.append(Source(
            title=title,
            url=None,
            snippet=doc.page_content[:300],
            source_type="rag",
        ))
    return sources


def format_rag_context(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    if not docs:
        return "No relevant private documents found."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        parts.append(f"[Document {i} — {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)