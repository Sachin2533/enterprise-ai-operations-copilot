
import hashlib
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from app.llm.general import (
    generate_general_answer
)
from app.rag.local_embeddings import (
    LocalHashEmbeddings
)

load_dotenv()

INDEX_DIR = "data/faiss_index"
DOCUMENTS_DIR = "data/documents"

EMBEDDING_MODEL = (
    "local-hash-embeddings"
)

FAISS_WEIGHT = 0.65
BM25_WEIGHT = 0.35

GENERIC_QUERY_TERMS = {
    "about",
    "details",
    "document",
    "information",
    "mention",
    "mentioned",
    "provided",
    "says",
    "what",
    "which",
}


def _document_response(
    answer: str,
    sources: list[dict] | None = None,
    confidence: str = "Low",
    retrieval: dict | None = None
) -> dict:

    return {
        "answer": answer,
        "metadata": {
            "sources": sources or [],
            "confidence": confidence,
            "retrieval": retrieval or {}
        }
    }


def _metadata_or_answer(
    response: dict,
    return_metadata: bool
):

    if return_metadata:

        return response

    return response["answer"]


def _not_found_response(
    detail: str,
    return_metadata: bool,
    sources: list[dict] | None = None,
    retrieval: dict | None = None
):

    return _metadata_or_answer(
        _document_response(
            answer=(
                f"I could not find this information in the selected document. "
                f"{detail}"
            ),
            sources=sources,
            confidence="Low",
            retrieval=retrieval
        ),
        return_metadata
    )


def _add_debug_event(
    debug_trace: list[dict] | None,
    step: str,
    detail: str,
    status: str = "ok"
):

    if debug_trace is None:

        return

    from datetime import datetime

    debug_trace.append(
        {
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "step": step,
            "detail": detail,
            "status": status
        }
    )


def _preferred_source_terms(question: str) -> list[str]:
    question = question.lower()

    if re.search(r"\b(resume|cv|candidate)\b", question):
        return [
            "resume",
            "cv"
        ]

    if re.search(r"\bhandbook\b", question):
        return [
            "handbook"
        ]

    if re.search(r"\b(policy|guidelines)\b", question):
        return [
            "policy",
            "guideline",
            "handbook"
        ]

    if re.search(r"\b(manual|guide)\b", question):
        return [
            "manual",
            "guide",
            "handbook"
        ]

    return []


def _tokenize(text: str) -> list[str]:

    return [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            text.lower()
        )
        if len(token) > 2
    ]


def _doc_key(doc) -> str:

    source = str(
        doc.metadata.get(
            "source",
            ""
        )
    )
    page = str(
        doc.metadata.get(
            "page",
            ""
        )
    )
    content_hash = hashlib.sha1(
        doc.page_content.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()

    return f"{source}|{page}|{content_hash}"


def _normalize_scores(
    scores: dict[str, float]
) -> dict[str, float]:

    if not scores:

        return {}

    values = list(
        scores.values()
    )
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:

        normalized_value = (
            1.0
            if maximum > 0
            else 0.0
        )

        return {
            key: normalized_value
            for key in scores
        }

    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in scores.items()
    }


def _bm25_scores(
    question: str,
    docs: list
) -> dict[str, float]:

    query_tokens = _tokenize(
        question
    )

    if not query_tokens or not docs:

        return {}

    tokenized_docs = [
        _tokenize(
            doc.page_content
        )
        for doc in docs
    ]

    average_length = (
        sum(
            len(tokens)
            for tokens in tokenized_docs
        )
        / max(
            len(tokenized_docs),
            1
        )
    )

    document_frequency = Counter()

    for tokens in tokenized_docs:

        document_frequency.update(
            set(tokens)
        )

    k1 = 1.5
    b = 0.75
    total_docs = len(docs)
    scores = {}

    for doc, tokens in zip(
        docs,
        tokenized_docs
    ):

        if not tokens:

            scores[_doc_key(doc)] = 0.0
            continue

        term_frequency = Counter(
            tokens
        )
        doc_length = len(tokens)
        score = 0.0

        for term in query_tokens:

            if term not in term_frequency:

                continue

            idf = math.log(
                1
                + (
                    total_docs
                    - document_frequency[term]
                    + 0.5
                )
                / (
                    document_frequency[term]
                    + 0.5
                )
            )
            frequency = term_frequency[term]
            denominator = (
                frequency
                + k1
                * (
                    1
                    - b
                    + b
                    * doc_length
                    / max(
                        average_length,
                        1
                    )
                )
            )
            score += idf * (
                frequency
                * (
                    k1
                    + 1
                )
                / denominator
            )

        scores[_doc_key(doc)] = score

    return scores


def _get_all_vectorstore_docs(vectorstore) -> list:

    raw_docs = getattr(
        vectorstore.docstore,
        "_dict",
        {}
    )

    return list(
        raw_docs.values()
    )


def _index_signature() -> str:

    index_files = [
        Path(INDEX_DIR) / "index.faiss",
        Path(INDEX_DIR) / "index.pkl"
    ]

    return "|".join(
        f"{path.name}:{path.stat().st_mtime_ns}"
        for path in index_files
        if path.exists()
    )


@lru_cache(maxsize=1)
def _load_embeddings():

    return LocalHashEmbeddings()


@lru_cache(maxsize=4)
def _load_vectorstore(
    signature: str
):

    from langchain_community.vectorstores import (
        FAISS
    )

    embeddings = _load_embeddings()

    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def _combine_hybrid_scores(
    docs_with_scores: list,
    all_docs: list,
    question: str,
    debug_trace: list[dict] | None = None
) -> list:

    faiss_distances = {
        _doc_key(doc): float(score)
        for doc, score in docs_with_scores
    }

    max_distance = max(
        faiss_distances.values(),
        default=1.0
    )

    faiss_relevance = {
        key: 1 - (distance / max(max_distance, 0.0001))
        for key, distance in faiss_distances.items()
    }

    bm25_raw = _bm25_scores(
        question,
        all_docs
    )
    bm25_normalized = _normalize_scores(
        bm25_raw
    )

    doc_lookup = {
        _doc_key(doc): doc
        for doc in all_docs
    }

    candidate_keys = set(
        faiss_relevance.keys()
    )

    candidate_keys.update(
        key
        for key, score in bm25_normalized.items()
        if score > 0
    )

    hybrid_results = []

    for key in candidate_keys:

        doc = doc_lookup.get(key)

        if doc is None:

            continue

        semantic_score = faiss_relevance.get(
            key,
            0.0
        )
        lexical_score = bm25_normalized.get(
            key,
            0.0
        )
        combined_score = (
            FAISS_WEIGHT
            * semantic_score
            + BM25_WEIGHT
            * lexical_score
        )

        hybrid_results.append(
            (
                doc,
                combined_score,
                semantic_score,
                lexical_score
            )
        )

    hybrid_results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    _add_debug_event(
        debug_trace,
        "hybrid_retrieval",
        (
            f"Combined FAISS ({FAISS_WEIGHT:.0%}) and BM25 "
            f"({BM25_WEIGHT:.0%}) across {len(hybrid_results)} candidates."
        )
    )

    return hybrid_results


def _source_matches_filter(
    source: str,
    source_filter: str | None
) -> bool:
    if not source_filter:
        return True

    return Path(source).name == Path(source_filter).name


def ask_document(
    question: str,
    source_filter: str | None = None,
    debug_trace: list[dict] | None = None,
    return_metadata: bool = False
):

    _add_debug_event(
        debug_trace,
        "rag.ask_document",
        "Started document retrieval."
    )

    if not Path(INDEX_DIR).exists():

        _add_debug_event(
            debug_trace,
            "rag.index_check",
            "FAISS index directory was missing.",
            status="warning"
        )

        if not any(Path(DOCUMENTS_DIR).glob("*.pdf")):

            return _metadata_or_answer(
                _document_response(
                    answer=(
                        "No document index is available yet. Upload a PDF "
                        "and rebuild the knowledge base before asking "
                        "document-specific questions."
                    )
                ),
                return_metadata
            )

        try:

            from app.rag.ingest import (
                build_vector_store
            )

            _add_debug_event(
                debug_trace,
                "build_vector_store",
                "Building FAISS index from PDFs."
            )

            build_vector_store()

        except Exception as exc:

            _add_debug_event(
                debug_trace,
                "build_vector_store",
                f"Failed to build document index: {exc}",
                status="error"
            )

            return _metadata_or_answer(
                _document_response(
                    answer=f"Unable to build the document index: {exc}"
                ),
                return_metadata
            )

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        _add_debug_event(
            debug_trace,
            "rag.api_key_check",
            "GROQ_API_KEY is missing.",
            status="error"
        )

        return _metadata_or_answer(
            _document_response(
                answer=(
                    "GROQ_API_KEY is not configured. Set it before asking "
                    "document questions."
                )
            ),
            return_metadata
        )

    try:

        embeddings = _load_embeddings()

        _add_debug_event(
            debug_trace,
            "LocalHashEmbeddings",
            f"Loaded/cached embedding model: {EMBEDDING_MODEL}"
        )

        vectorstore = _load_vectorstore(
            _index_signature()
        )

        _add_debug_event(
            debug_trace,
            "FAISS.load_local",
            f"Loaded/cached vector index from {INDEX_DIR}."
        )

        docs_with_scores = vectorstore.similarity_search_with_score(
            question,
            k=100 if source_filter else 40
        )

        _add_debug_event(
            debug_trace,
            "similarity_search_with_score",
            f"Retrieved {len(docs_with_scores)} semantic candidate chunks."
        )

        all_docs = _get_all_vectorstore_docs(
            vectorstore
        )

    except Exception as exc:

        _add_debug_event(
            debug_trace,
            "rag.search",
            f"Unable to search document index: {exc}",
            status="error"
        )

        return _metadata_or_answer(
            _document_response(
                answer=f"Unable to search the document index: {exc}"
            ),
            return_metadata
        )

    if not docs_with_scores:

        return _not_found_response(
            "Try asking with a more specific phrase from the document.",
            return_metadata
        )

    if source_filter:

        docs_with_scores = [
            (doc, score)
            for doc, score in docs_with_scores
            if _source_matches_filter(
                str(
                    doc.metadata.get(
                        "source",
                        ""
                    )
                ),
                source_filter
            )
        ]

        all_docs = [
            doc
            for doc in all_docs
            if _source_matches_filter(
                str(
                    doc.metadata.get(
                        "source",
                        ""
                    )
                ),
                source_filter
            )
        ]

        _add_debug_event(
            debug_trace,
            "source_filter",
            f"Filtered candidates to {len(docs_with_scores)} chunks for {Path(source_filter).name}."
        )

        if not docs_with_scores:

            return _not_found_response(
                (
                    f"I could not find indexed content for "
                    f"{Path(source_filter).name}. Rebuild the knowledge base "
                    "and try again."
                ),
                return_metadata
            )

    hybrid_results = _combine_hybrid_scores(
        docs_with_scores,
        all_docs,
        question,
        debug_trace=debug_trace
    )

    if not hybrid_results:

        return _not_found_response(
            "Try a more specific document question.",
            return_metadata
        )

    preferred_terms = _preferred_source_terms(question)

    if preferred_terms and not source_filter:

        preferred_results = [
            item
            for item in hybrid_results
            if any(
                term in str(
                    item[0].metadata.get(
                        "source",
                        ""
                    )
                ).lower()
                for term in preferred_terms
            )
        ]

        if preferred_results:

            hybrid_results = preferred_results

            _add_debug_event(
                debug_trace,
                "preferred_source_terms",
                f"Preferred sources matched terms: {', '.join(preferred_terms)}."
            )

    selected_results = hybrid_results[:3]

    _add_debug_event(
        debug_trace,
        "context_selection",
        f"Selected {len(selected_results)} hybrid-ranked chunks for the LLM context."
    )

    _, best_score, _, _ = selected_results[0]
    docs = [
        doc
        for doc, _, _, _ in selected_results
    ]

    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if len(term) > 2
        and term not in GENERIC_QUERY_TERMS
    }

    doc_text = " ".join(
        doc.page_content.lower()
        for doc in docs
    )

    overlap = sum(
        1
        for term in query_terms
        if term in doc_text
    )

    relevance_ratio = (
        overlap / max(len(query_terms), 1)
    )

    is_summary_question = bool(
        re.search(
            r"\b(summarize|summary|overview|key points)\b",
            question,
            re.IGNORECASE
        )
    )

    retrieval_metadata = {
        "hybrid_score": round(
            best_score,
            3
        ),
        "relevance_ratio": round(
            relevance_ratio,
            3
        ),
        "chunks_selected": len(docs),
        "retrieval_type": "hybrid_bm25_faiss"
    }

    if (
        not is_summary_question
        and (
            (
                len(query_terms) >= 2
                and overlap < 2
            )
            or
            relevance_ratio < 0.15
            or (
                best_score < 0.18
                and relevance_ratio < 0.25
            )
        )
    ):

        _add_debug_event(
            debug_trace,
            "relevance_guard",
            f"Rejected weak match. hybrid_score={best_score:.2f}, relevance_ratio={relevance_ratio:.2f}",
            status="warning"
        )

        return _not_found_response(
            "The retrieved chunks were too weak to answer reliably.",
            return_metadata,
            retrieval=retrieval_metadata
        )

    context_blocks = []
    source_references = []
    structured_sources = []

    for index, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "uploaded document"
        )
        page = doc.metadata.get("page")
        source_name = Path(
            str(source)
        ).name
        citation = (
            f"Source {index}: {source_name}"
        )

        if page is not None:

            citation += f", page {page + 1}"
            source_references.append(
                f"Source {index}: {source_name}, page {page + 1}"
            )
            structured_sources.append(
                {
                    "source_number": index,
                    "file": source_name,
                    "page": page + 1
                }
            )

        else:

            source_references.append(
                f"Source {index}: {source_name}"
            )
            structured_sources.append(
                {
                    "source_number": index,
                    "file": source_name,
                    "page": None
                }
            )

        context_blocks.append(
            f"{citation}\n{doc.page_content}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are an Enterprise AI Operations Copilot.
Use ONLY the context below for document-specific claims.
If the context does not answer the question, say that clearly.
Never answer from general knowledge for document questions.
Never add facts that are not present in the context.
Do not use a fixed report template.
Start with the direct answer in clear, simple language.
Use short bullets when there are multiple points.
Include source references using the Source numbers provided.
For every document-specific fact, cite the source inline like (Source 1, page 3).
End with a short "Sources" line listing the source numbers and page numbers used.
Add a brief next step only when it is genuinely useful.

Context:
{context}

Question:
{question}
"""

    try:

        from langchain_groq import ChatGroq

        llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile"
        )

        _add_debug_event(
            debug_trace,
            "ChatGroq.invoke",
            "Calling Groq LLM with retrieved document context."
        )

        response = llm.invoke(
            prompt
        )

    except Exception as exc:

        _add_debug_event(
            debug_trace,
            "ChatGroq.invoke",
            f"Document answer generation failed: {exc}",
            status="error"
        )

        return _metadata_or_answer(
            _document_response(
                answer=f"Unable to answer from the document context: {exc}",
                sources=structured_sources,
                confidence="Low",
                retrieval=retrieval_metadata
            ),
            return_metadata
        )

    _add_debug_event(
        debug_trace,
        "rag.answer",
        "Document answer generated successfully."
    )

    answer = response.content

    unsupported_markers = [
        "general knowledge",
        "not widely recognized",
        "outside the provided context",
        "not in the provided context",
    ]

    if any(
        marker in answer.lower()
        for marker in unsupported_markers
    ):

        return _not_found_response(
            "The selected document did not provide enough supporting context.",
            return_metadata,
            sources=structured_sources,
            retrieval=retrieval_metadata
        )

    if (
        source_references
        and "sources:" not in answer.lower()
    ):

        answer = (
            answer.rstrip()
            + "\n\nSources: "
            + "; ".join(source_references)
        )

    if best_score >= 0.45 and relevance_ratio >= 0.35:

        confidence = "High"

    elif best_score >= 0.22 or relevance_ratio >= 0.25:

        confidence = "Medium"

    else:

        confidence = "Low"

    return _metadata_or_answer(
        _document_response(
            answer=answer,
            sources=structured_sources,
            confidence=confidence,
            retrieval=retrieval_metadata
        ),
        return_metadata
    )
