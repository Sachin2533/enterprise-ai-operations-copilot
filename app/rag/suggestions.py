import re
from collections import Counter
from pathlib import Path

from pypdf import (
    PdfReader
)


DOCUMENTS_DIR = Path(
    "data/documents"
)

STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "can",
    "could",
    "does",
    "each",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "should",
    "such",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
    "would",
}

NOISY_PHRASES = {
    "by",
    "page",
    "date",
    "name",
    "email",
    "phone",
    "address",
    "summary",
    "objective",
}

DOMAIN_TERMS = {
    "approval",
    "approvals",
    "architecture",
    "benefits",
    "candidate",
    "clauses",
    "compliance",
    "course",
    "duration",
    "deadline",
    "diploma",
    "eligibility",
    "experience",
    "fee",
    "fees",
    "features",
    "findings",
    "institute",
    "implementation",
    "limitations",
    "methodology",
    "objectives",
    "obligations",
    "penalties",
    "policy",
    "process",
    "project",
    "projects",
    "qualification",
    "requirements",
    "responsibilities",
    "risks",
    "skills",
    "steps",
    "technologies",
    "timeline",
}


def _document_path(filename: str) -> Path:

    return DOCUMENTS_DIR / Path(
        filename
    ).name


def _load_document_text(filename: str) -> str:

    path = _document_path(
        filename
    )

    if not path.exists():

        return ""

    try:

        reader = PdfReader(
            str(path)
        )

    except Exception:

        return ""

    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


def _clean_phrase(phrase: str) -> str:

    phrase = re.sub(
        r"\s+",
        " ",
        phrase
    ).strip(" -:|,.")

    return phrase


def _extract_heading_candidates(text: str) -> list[str]:

    headings = []

    for line in text.splitlines():

        cleaned = _clean_phrase(
            line
        )

        if not cleaned:

            continue

        if cleaned.lower() in NOISY_PHRASES:

            continue

        words = cleaned.split()

        if len(words) < 2 or len(words) > 8:

            continue

        if cleaned.isupper() or cleaned.istitle():

            headings.append(
                cleaned
            )

    return headings


def _extract_keywords(text: str) -> list[str]:

    tokens = [
        token.lower()
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9+#.-]{2,}",
            text
        )
    ]

    filtered_tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
        and not token.isdigit()
    ]

    counts = Counter(
        filtered_tokens
    )

    scored_terms = sorted(
        counts.items(),
        key=lambda item: (
            item[0] in DOMAIN_TERMS,
            item[1],
            len(item[0])
        ),
        reverse=True
    )

    return [
        term
        for term, _ in scored_terms[:12]
    ]


def _extract_named_phrases(text: str) -> list[str]:

    phrases = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9+#.-]+(?:\s+|$)){2,5}",
        text
    )

    cleaned_phrases = []

    for phrase in phrases:

        cleaned = _clean_phrase(
            phrase
        )

        if (
            cleaned
            and len(cleaned.split()) <= 5
            and cleaned.lower() not in STOPWORDS
            and cleaned.lower() not in NOISY_PHRASES
        ):

            cleaned_phrases.append(
                cleaned
            )

    counts = Counter(
        cleaned_phrases
    )

    return [
        phrase
        for phrase, _ in counts.most_common(8)
    ]


def _dedupe_questions(questions: list[str]) -> list[str]:

    seen = set()
    subjects = set()
    deduped = []

    for question in questions:

        normalized = question.lower()

        subject_match = re.search(
            r"(?:about|provided about) (.+?)\?",
            question,
            re.IGNORECASE
        )

        if subject_match:

            subject = subject_match.group(1).lower()

            if subject in subjects:

                continue

            subjects.add(
                subject
            )

        if normalized in seen:

            continue

        seen.add(
            normalized
        )
        deduped.append(
            question
        )

    return deduped[:6]


def _normalize_text(text: str) -> str:

    return " ".join(
        re.findall(
            r"[a-z0-9+#.-]+",
            text.lower()
        )
    )


def _has_evidence(
    normalized_text: str,
    terms: list[str],
    minimum_matches: int = 1
) -> bool:

    matches = 0

    for term in terms:

        term = term.lower()

        if term in normalized_text:

            matches += 1

    return matches >= minimum_matches


def _add_candidate(
    candidates: list[tuple[str, list[str], int]],
    question: str,
    terms: list[str],
    minimum_matches: int = 1
):

    candidates.append(
        (
            question,
            terms,
            minimum_matches
        )
    )


def _filter_answerable_questions(
    candidates: list[tuple[str, list[str], int]],
    text: str
) -> list[str]:

    normalized_text = _normalize_text(
        text
    )
    answerable_questions = []

    for question, terms, minimum_matches in candidates:

        if _has_evidence(
            normalized_text,
            terms,
            minimum_matches
        ):

            answerable_questions.append(
                question
            )

    return _dedupe_questions(
        answerable_questions
    )


def get_suggested_questions(filename: str) -> list[str]:

    text = _load_document_text(
        filename
    )

    if not text.strip():

        return []

    headings = _extract_heading_candidates(
        text
    )
    keywords = _extract_keywords(
        text
    )
    named_phrases = _extract_named_phrases(
        text
    )

    candidates: list[tuple[str, list[str], int]] = []

    _add_candidate(
        candidates,
        "Summarize this document using only the uploaded content.",
        _extract_keywords(text)[:5],
        minimum_matches=2
    )

    _add_candidate(
        candidates,
        "What are the key points and important details in this document?",
        _extract_keywords(text)[:5],
        minimum_matches=2
    )

    if re.search(
        r"\b(course|diploma|fee|fees|admission|eligibility|qualification|career)\b",
        text,
        re.IGNORECASE
    ):

        _add_candidate(
            candidates,
            "What is the eligibility or qualification required?",
            [
                "eligibility",
                "qualification",
                "required",
                "admission"
            ],
            minimum_matches=1
        )

        _add_candidate(
            candidates,
            "What course duration, fees, or admission details are mentioned?",
            [
                "course",
                "duration",
                "fee",
                "fees",
                "admission"
            ],
            minimum_matches=2
        )

        _add_candidate(
            candidates,
            "What career options or outcomes are listed?",
            [
                "career",
                "job",
                "pharmacist",
                "outcome",
                "opportunities"
            ],
            minimum_matches=1
        )

    for heading in headings[:2]:

        _add_candidate(
            candidates,
            f"What does the document say about {heading}?",
            _extract_keywords(heading) or [heading.lower()],
            minimum_matches=1
        )

    for phrase in named_phrases[:2]:

        _add_candidate(
            candidates,
            f"What details are provided about {phrase}?",
            _extract_keywords(phrase) or [phrase.lower()],
            minimum_matches=1
        )

    for keyword in keywords[:4]:

        _add_candidate(
            candidates,
            f"What does this document mention about {keyword}?",
            [
                keyword
            ],
            minimum_matches=1
        )

    if re.search(
        r"\b(skill|skills|python|java|sql|project|experience)\b",
        text,
        re.IGNORECASE
    ):

        _add_candidate(
            candidates,
            "What skills, projects, and experience are described?",
            [
                "skills",
                "projects",
                "experience"
            ],
            minimum_matches=2
        )

    if re.search(
        r"\b(policy|approval|eligibility|compliance|rule|deadline)\b",
        text,
        re.IGNORECASE
    ):

        _add_candidate(
            candidates,
            "What rules, approvals, deadlines, or compliance points are mentioned?",
            [
                "rules",
                "approval",
                "approvals",
                "deadline",
                "compliance"
            ],
            minimum_matches=1
        )

    if re.search(
        r"\b(risk|penalty|limitation|exception|caution|warning)\b",
        text,
        re.IGNORECASE
    ):

        _add_candidate(
            candidates,
            "What risks, limitations, penalties, or exceptions are mentioned?",
            [
                "risk",
                "risks",
                "limitations",
                "penalties",
                "exceptions"
            ],
            minimum_matches=1
        )

    return _filter_answerable_questions(
        candidates,
        text
    )
