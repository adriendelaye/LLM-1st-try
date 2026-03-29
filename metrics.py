# metrics.py
from sklearn.metrics import pairwise_distances
import numpy as np

class Metrics:
    @staticmethod
    def recall_at_k(retrieved_docs, relevant_docs, k=5):
        """
        Calcul du Recall@k
        retrieved_docs : liste de docs récupérés (ou ids)
        relevant_docs : liste de docs pertinents (ou ids)
        k : top-k à considérer
        """
        top_k = retrieved_docs[:k]
        hits = sum(1 for doc in top_k if doc in relevant_docs)
        return hits / len(relevant_docs) if relevant_docs else 0.0

    @staticmethod
    def semantic_coherence(docs_embeddings):
        """
        Cohérence sémantique : similarité moyenne cosine entre tous les chunks sélectionnés
        docs_embeddings : numpy array shape (n_docs, embedding_dim)
        """
        if len(docs_embeddings) <= 1:
            return 1.0
        cosine_dist = pairwise_distances(docs_embeddings, metric="cosine")
        # diagonale = 0, on prend la moyenne hors diagonale
        n = len(docs_embeddings)
        mean_cosine_similarity = 1 - (np.sum(cosine_dist) / (n*(n-1)))
        return mean_cosine_similarity

    @staticmethod
    def linguistic_precision(generated_text, reference_texts):
        """
        Précision linguistique simple : ratio de mots communs avec les références
        """
        gen_tokens = set(generated_text.lower().split())
        ref_tokens = set()
        for ref in reference_texts:
            ref_tokens.update(ref.lower().split())
        if not ref_tokens:
            return 0.0
        common = gen_tokens.intersection(ref_tokens)
        return len(common) / len(ref_tokens)
