# reranker.py
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        self._cache = {}

    def prepare_pairs(self, query, documents):
        return [(query, doc["text"]) for doc in documents]

    def rerank(self, query, documents, top_k=5):
        pairs_to_compute = []
        indices_to_compute = []

        # Cache check
        scores = np.zeros(len(documents))
        for i, doc in enumerate(documents):
            key = (query, doc["text"])
            if key in self._cache:
                scores[i] = self._cache[key]
            else:
                pairs_to_compute.append(key)
                indices_to_compute.append(i)

        # Batching
        if pairs_to_compute:
            batch_scores = self.model.predict(pairs_to_compute, batch_size=self.batch_size)
            for idx, score in zip(indices_to_compute, batch_scores):
                scores[idx] = score
                for pair, idx, score in zip(pairs_to_compute, indices_to_compute, batch_scores):
                    scores[idx] = score
                    self._cache[pair] = score

        # Ranking
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:top_k]]
