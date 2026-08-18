"""Indexación del código: chunking + vector store.

Pasos (todos deterministas, sin LLM):

1. **Chunking**: dividir cada archivo en fragmentos (`Document`) que caben
   en la ventana de contexto del modelo y conservan metadatos (ruta,
   lenguaje, líneas) para poder citar la fuente en las respuestas.
2. **Embeddings**: convertir cada chunk en un vector numérico.
3. **Vector store**: guardar los vectores para poder buscar por similitud.

Usamos `InMemoryVectorStore` de LangChain: es un vector store real
(almacena vectores y hace búsqueda por similitud coseno) pero vive en
memoria, sin necesidad de un servidor externo como Chroma o FAISS.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_code_analyzer.repo import FileInfo


def split_files(files: list[FileInfo], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Divide los archivos en `Document`s listos para indexar."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    documents: list[Document] = []
    for file in files:
        # Incluir la ruta y el lenguaje dentro del contenido ayuda al
        # retriever a emparejar preguntas con los archivos adecuados.
        page_content = f"FILE: {file.path}\nLANGUAGE: {file.language}\n\n{file.content}"

        chunks = splitter.split_text(page_content)
        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file.path,
                        "language": file.language,
                        "chunk": index,
                    },
                )
            )
    return documents


def build_vectorstore(documents: list[Document], embeddings):
    """Crea el vector store a partir de los chunks y el motor de embeddings."""
    return InMemoryVectorStore.from_documents(documents, embeddings)


def format_context_doc(doc: Document, max_chars: int = 3000) -> str:
    """Formatea un documento recuperado para incluirlo en el prompt del LLM."""
    source = doc.metadata.get("source", "desconocido")
    language = doc.metadata.get("language", "")
    content = doc.page_content
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... (truncado)"
    return f"--- {source} ({language}) ---\n{content}"
