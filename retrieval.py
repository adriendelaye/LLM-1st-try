# retrieval.py
    
import os
import faiss
import numpy as np
from typing import List, Dict
import pickle
from rank_bm25 import BM25Okapi
from reranker import Reranker

class FaissRetriever:
    """
    FAISS-based semantic retriever using HNSW index.
    Supports cosine similarity via L2 normalization.
    """

    def __init__(self, d: int = 384, hnsw_m: int = 32, ef_search: int = 64):
        self.d = d
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.bm25 = None
        self.tokenized_corpus = None
        self.reranker = None

        self.index = self._create_index()
        self.docs: List[Dict[str, any]] = []

    def _create_index(self):
        index = faiss.IndexHNSWFlat(self.d, self.hnsw_m)
        index.hnsw.efSearch = self.ef_search
        return index

    def build_index(self, embeddings: np.ndarray, docs: List[Dict[str, any]]) -> None:
        """
        Build FAISS index from embeddings and associated documents.
        """
        embeddings = np.array(embeddings, dtype="float32")

        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(f"Expected embeddings shape (n, {self.d})")

        if len(embeddings) != len(docs):
            raise ValueError("Embeddings and docs size mismatch")

        # Reset index (important!)
        self.index = self._create_index()

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.docs = docs
        self._build_bm25()

    def _build_bm25(self):
        corpus = [doc["text"] for doc in self.docs]
        tokenized = [text.lower().split() for text in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.tokenized_corpus = tokenized

    def search(self, query_emb: np.ndarray, k: int = 30) -> List[Dict]:
        """
        Dense retrieval with scores and ranks.
        """
        if self.index.ntotal == 0:
            return []

        query_emb = np.array(query_emb, dtype="float32")

        if query_emb.ndim != 2 or query_emb.shape[1] != self.d:
            raise ValueError(f"Query embedding must be shape (1, {self.d})")

        query_emb = query_emb.copy()
        faiss.normalize_L2(query_emb)

        k = min(k, len(self.docs))
        if k <= 0:
            return []

        try:
            D, I = self.index.search(query_emb, k)
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {e}")

        results = []

        for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx == -1:
                continue

            score = 1 - (dist ** 2) / 2

            doc = self.docs[idx].copy()
            doc["score"] = float(score)
            doc["rank"] = rank + 1

            results.append(doc)

        return results

    def dense_search(self, query_emb: np.ndarray, k: int = 30):
        query_emb = query_emb.copy()
        faiss.normalize_L2(query_emb)

        D, I = self.index.search(query_emb, k)

        return [(idx, dist) for idx, dist in zip(I[0], D[0]) if idx != -1]

    def bm25_search(self, query: str, k: int = 30):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_k = np.argsort(scores)[::-1][:k]

        return [(idx, scores[idx]) for idx in top_k]
    
    def hybrid_search(self, query: str, query_emb: np.ndarray, k: int = 5):

        dense = self.dense_search(query_emb, k=30)
        sparse = self.bm25_search(query, k=30)

        fused = self.reciprocal_rank_fusion(dense, sparse)

        # reranker
        if self.reranker:
            docs = [self.docs[idx] for idx, _ in fused]
            return self.reranker.rerank(query, docs, top_k=k)

        # fallback
        results = []
        for rank, (idx, score) in enumerate(fused[:k]):
            doc = self.docs[idx].copy()
            doc["fusion_score"] = float(score)
            doc["rank"] = rank + 1
            results.append(doc)

        return results

    def reciprocal_rank_fusion(self, dense, sparse, k=60):
        scores = {}

        for rank, (idx, _) in enumerate(dense):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

        for rank, (idx, _) in enumerate(sparse):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def save(self, path: str) -> None:
        """
        Save FAISS index and documents to disk.
        """
        if self.index.ntotal == 0:
            raise ValueError("Cannot save empty index")

        os.makedirs(path, exist_ok=True)

        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "docs.pkl")

        try:
            faiss.write_index(self.index, index_path)
            with open(docs_path, "wb") as f:
                pickle.dump(self.docs, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save index: {e}")
    def load(self, path: str) -> None:
        """
        Load FAISS index and documents from disk.
        """
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "docs.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing FAISS index at {index_path}")

        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Missing docs file at {docs_path}")

        try:
            self.index = faiss.read_index(index_path)

            with open(docs_path, "rb") as f:
                self.docs = pickle.load(f)

        except Exception as e:
            raise RuntimeError(f"Failed to load index: {e}")

        # Validation
        if self.index.ntotal != len(self.docs):
            raise ValueError("Index size and docs size mismatch after loading")
        self._build_bm25()
        
    def load_reranker(self):
        self.reranker = Reranker()
