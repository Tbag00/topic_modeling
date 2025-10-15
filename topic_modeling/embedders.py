"""Custom embedder wrappers for BERTopic."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pandas as pd

class NormalizedSentenceTransformer(BaseEmbedder):
    """SentenceTransformer wrapper that returns L2-normalized embeddings."""

    def __init__(
        self,
        model_name: str,
        mean: bool = False,
        device: Optional[str] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        normalize_embeddings: bool = True,
        text_column: str = "Description",
        group_column: str = "des_id",
        paragraph_order_column: Optional[str] = "par_id",
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs or {}
        self.normalize_embeddings = normalize_embeddings
        self.text_column = text_column
        self.group_column = group_column
        self.paragraph_order_column = paragraph_order_column
        self.aggregate_descriptions = mean
        self.mean = mean  # backwards compatibility with earlier code
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                **self.model_kwargs,
            )
        return self._model

    def embed(self, documents: Any, verbose: bool = False) -> np.ndarray:
        encode_opts = {"show_progress_bar": verbose, **self.encode_kwargs}
        if self.aggregate_descriptions:
            paragraphs, doc_lengths = self._collect_paragraphs(documents)
            if not paragraphs:
                dim = self.model.get_sentence_embedding_dimension()
                embeddings = np.empty((0, dim * 2), dtype=np.float32)
            else:
                paragraph_embeddings = self.model.encode(paragraphs, **encode_opts)
                paragraph_embeddings = normalize(
                    np.asarray(paragraph_embeddings, dtype=np.float32)
                ).astype(np.float32, copy=False)
                embeddings = self._aggregate_paragraph_embeddings(
                    paragraph_embeddings,
                    doc_lengths,
                )
        else:
            texts = self._collect_texts(documents)
            if not texts:
                dim = self.model.get_sentence_embedding_dimension()
                embeddings = np.empty((0, dim), dtype=np.float32)
            else:
                embeddings = self.model.encode(texts, **encode_opts)

        if self.normalize_embeddings:
            embeddings = normalize(embeddings).astype(np.float32, copy=False)
        return embeddings

    def _collect_texts(self, documents: Any) -> List[str]:
        if isinstance(documents, pd.DataFrame):
            if self.text_column not in documents.columns:
                raise ValueError(
                    f"Missing column '{self.text_column}' in provided DataFrame."
                )
            return documents[self.text_column].astype(str).tolist()
        if isinstance(documents, pd.Series):
            return documents.astype(str).tolist()
        if isinstance(documents, (list, tuple)):
            return [str(doc) for doc in documents]
        raise TypeError(
            "Documents must be a pandas DataFrame, Series, or a sequence of strings."
        )

    def _collect_paragraphs(self, documents: Any) -> Tuple[List[str], List[int]]:
        if isinstance(documents, pd.DataFrame) and self.group_column in documents.columns:
            df = documents
            if self.text_column not in df.columns:
                raise ValueError(
                    f"Missing column '{self.text_column}' in provided DataFrame."
                )
            paragraphs: List[str] = []
            lengths: List[int] = []
            unique_ids = pd.unique(df[self.group_column])
            for group_id in unique_ids:
                group = df.loc[df[self.group_column] == group_id]
                if (
                    self.paragraph_order_column
                    and self.paragraph_order_column in group.columns
                ):
                    group = group.sort_values(self.paragraph_order_column)
                texts = [
                    text.strip()
                    for text in group[self.text_column].astype(str).tolist()
                    if text.strip()
                ]
                if not texts:
                    texts = [""]
                lengths.append(len(texts))
                paragraphs.extend(texts)
            return paragraphs, lengths

        texts = self._collect_texts(documents)
        return self._split_documents_into_paragraphs(texts)

    def _split_documents_into_paragraphs(
        self, documents: Sequence[str]
    ) -> Tuple[List[str], List[int]]:
        paragraphs: List[str] = []
        lengths: List[int] = []
        for doc in documents:
            splits = [part.strip() for part in doc.split("\n") if part.strip()]
            if not splits:
                stripped = doc.strip()
                splits = [stripped] if stripped else [""]
            lengths.append(len(splits))
            paragraphs.extend(splits)
        return paragraphs, lengths

    def _aggregate_paragraph_embeddings(
        self, paragraph_embeddings: np.ndarray, doc_lengths: Sequence[int]
    ) -> np.ndarray:
        if not doc_lengths:
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim * 2), dtype=np.float32)

        embedding_dim = paragraph_embeddings.shape[1]
        aggregated = np.empty(
            (len(doc_lengths), embedding_dim * 2), dtype=np.float32
        )
        start = 0
        for idx, length in enumerate(doc_lengths):
            end = start + length
            doc_embeddings = paragraph_embeddings[start:end]
            mean_vec = doc_embeddings.mean(axis=0)
            max_vec = doc_embeddings.max(axis=0)
            aggregated[idx, :embedding_dim] = mean_vec
            aggregated[idx, embedding_dim:] = max_vec
            start = end
        return aggregated

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self._model = None