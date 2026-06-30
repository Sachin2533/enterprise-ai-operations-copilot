import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import (
    Embeddings
)


class LocalHashEmbeddings(Embeddings):
    """
    Small deterministic embedding backend for local/offline RAG.
    It avoids sentence-transformers imports while keeping FAISS usable.
    """

    def __init__(
        self,
        dimension: int = 384
    ):

        self.dimension = dimension

    def _embed(
        self,
        text: str
    ) -> list[float]:

        vector = [
            0.0
            for _ in range(self.dimension)
        ]
        tokens = re.findall(
            r"[a-z0-9]+",
            text.lower()
        )

        if not tokens:

            return vector

        counts = Counter(
            token
            for token in tokens
            if len(token) > 2
        )

        for token, count in counts.items():

            digest = hashlib.sha1(
                token.encode("utf-8")
            ).digest()
            index = int.from_bytes(
                digest[:4],
                "big"
            ) % self.dimension
            sign = (
                1.0
                if digest[4] % 2 == 0
                else -1.0
            )
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(
            sum(
                value * value
                for value in vector
            )
        )

        if norm == 0:

            return vector

        return [
            value / norm
            for value in vector
        ]

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str
    ) -> list[float]:

        return self._embed(text)
