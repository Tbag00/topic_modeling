"""Custom embedder wrappers for BERTopic."""

from typing import Any, Dict, List, Optional

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
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs or {}
        self.normalize_embeddings = normalize_embeddings
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

    def embed(self, df: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        
        encode_opts = {"show_progress_bar": verbose, **self.encode_kwargs}
        embeddings = []
        if self.mean:
            
        embeddings = self.model.encode(documents, **encode_opts)
        if self.normalize_embeddings:
            embeddings = normalize(embeddings)
        return embeddings

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self._model = None
