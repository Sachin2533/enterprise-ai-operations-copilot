from pathlib import Path
import shutil

from langchain_core.documents import (
    Document
)

from pypdf import (
    PdfReader
)

from app.rag.local_embeddings import (
    LocalHashEmbeddings
)


DOCUMENTS_DIR = "data/documents"
INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL = "local-hash-embeddings"


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[Document]:

    chunks = []
    step = max(
        chunk_size - chunk_overlap,
        1
    )

    for document in documents:

        text = document.page_content or ""

        for start in range(
            0,
            len(text),
            step
        ):

            chunk_text = text[
                start:start + chunk_size
            ].strip()

            if not chunk_text:

                continue

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=dict(
                        document.metadata
                    )
                )
            )

    return chunks


def load_pdf_documents(
    pdf_path: Path
) -> list[Document]:

    reader = PdfReader(
        str(pdf_path)
    )
    documents = []

    for page_index, page in enumerate(
        reader.pages
    ):

        text = page.extract_text() or ""

        if not text.strip():

            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_index
                }
            )
        )

    return documents


def build_vector_store():

    documents = []

    pdf_files = Path(
        DOCUMENTS_DIR
    ).glob("*.pdf")

    for pdf in pdf_files:

        documents.extend(
            load_pdf_documents(
                pdf
            )
        )

    if not documents:

        raise Exception(
            "No PDF files found."
        )

    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    from langchain_community.vectorstores import (
        FAISS
    )

    embeddings = LocalHashEmbeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        INDEX_DIR
    )

    print(
        f"Indexed {len(chunks)} chunks."
    )


def clear_vector_store():

    index_path = Path(
        INDEX_DIR
    )

    if index_path.exists():

        shutil.rmtree(
            index_path
        )


def refresh_vector_store():

    if any(
        Path(DOCUMENTS_DIR).glob("*.pdf")
    ):

        build_vector_store()

    else:

        clear_vector_store()


def delete_document(
    document_name: str
):

    safe_name = Path(
        document_name
    ).name
    document_path = Path(
        DOCUMENTS_DIR
    ) / safe_name

    if not document_path.exists():

        raise FileNotFoundError(
            f"{safe_name} was not found."
        )

    document_path.unlink()

    refresh_vector_store()

    return safe_name


if __name__ == "__main__":

    build_vector_store()
