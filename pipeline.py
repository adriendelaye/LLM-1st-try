# pipeline.py

from embedding import Embedder
from retrieval import FaissRetriever
from reranker import Reranker
from generator import LLMGenerator
from prompting import build_context, build_prompt
import numpy as np

class AdvancedRAGPipeline: # configurable parameters
    def __init__(
        self,
        embedder: Embedder = None,
        retriever: FaissRetriever = None,
        reranker: Reranker = None,
        generator: LLMGenerator = None,
        top_k_retrieval: int = 30,
        top_k_rerank: int = 5,
        max_chunks: int = 3,
        domain_filter: str = None,
        batch_size_embed: int = 32,
        batch_size_rerank: int = 16
    ):
        # Dependances injection
        self.embedder = embedder or Embedder()
        self.retriever = retriever or FaissRetriever()
        self.reranker = reranker or Reranker(batch_size=batch_size_rerank)
        HF_TOKEN = "" #put in your HuggingFace Token
        self.generator = generator or LLMGenerator(model="mistral", hf_token = HF_TOKEN)
        
        # Parameters to be called inside
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.max_chunks = max_chunks
        self.domain_filter = domain_filter
        self.batch_size_embed = batch_size_embed
        self.batch_size_rerank = batch_size_rerank

        # Link reranker to retriever (option for hybrid_search)
        self.retriever.reranker = self.reranker

    def select_chunks(self, query_emb, reranked_docs, embeddings_dict, alpha=0.6, beta=0.4):
        """
        Advanced selected chunks with ponderate rerank + cosine similarity.
        
        query_emb       : query embedding (numpy array 1,d)
        reranked_docs   : list of reranked docc [{text, concept, rerank_score, ...}]
        embeddings_dict : dict {doc_id: embedding} calculate cosine similarity
        alpha           : weight of rerank score
        beta            : weight of cosine score 
        """
        selected = []
        seen_concepts = set()
        scores = []

        for doc in reranked_docs:
            doc_id = doc.get("id", doc.get("text"))
            rerank_score = doc.get("rerank_score", 1.0)
            chunk_emb = embeddings_dict.get(doc_id)
            if chunk_emb is not None:
                cosine_sim = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-8)
            else:
                cosine_sim = 0.0
            combined_score = alpha * rerank_score + beta * cosine_sim
            scores.append((doc, combined_score))

        # sort by combined score
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

        for doc, _ in scores_sorted:
            concept = doc.get("concept")
            if concept not in seen_concepts:
                selected.append(doc)
                seen_concepts.add(concept)
            if len(selected) >= self.max_chunks:
                break

        return selected

    def __call__(self, query, domain_filter=None):
        """
        Complete Pipeline :
        1. Embedding
        2. Hybrid retrieval (dense + BM25 + RRF)
        3. Filter per domain
        4. Rerank
        5. Selection of chunks
        6. Construction of context and prompt
        7. LLM generation
        """
        # Query embedding
        query_emb = self.embedder.embed_texts([query], batch_size=self.batch_size_embed)

        # Hybrid retrieval (dense + BM25 + RRF)
        retrieved = self.retriever.hybrid_search(query, query_emb, k=self.top_k_retrieval)

        # Filter per domain
        domain_to_use = domain_filter or self.domain_filter
        if domain_to_use:
            filtered = [doc for doc in retrieved if doc.get("domain","") == domain_to_use]
            if not filtered:
                filtered = retrieved
        else:
            filtered = retrieved

        # Reranking (already applied in hybrid_search if reranker exists)
        # we can rerank again if necessary
        reranked = self.reranker.rerank(query, filtered, top_k=self.top_k_rerank)

        # Select chunks
        embeddings_dict = {doc.get("id", doc["text"]): doc.get("embedding") for doc in filtered}

        # Flatten query embedding (1,d)
        query_emb_flat = query_emb[0]

        selected = self.select_chunks(query_emb_flat, reranked, embeddings_dict)

        # builds context
        context = build_context(selected)

        # builds prompt
        prompt = build_prompt(query, context)

        # Generate LLM
        answer = self.generator.generate([prompt])[0]

        return answer


# Wrapper safe pour test/debug
class SafeAdvancedRAGPipeline(AdvancedRAGPipeline):
    def __call__(self, query, domain_filter=None):
        try:
            return super().__call__(query, domain_filter)
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
            return "[ERROR] Pipeline failed."
