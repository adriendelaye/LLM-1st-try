# evaluation.py
from metrics import Metrics
import numpy as np

def evaluate_pipeline(query, pipeline, docs, top_k=5):
    # 1. Appel du pipeline
    generated_answer = pipeline(query)

    # 2. Recall@k sur les documents sélectionnés
    retrieved_docs = [doc["text"] for doc in docs]
    relevant_docs = [doc["text"] for doc in docs if doc.get("is_relevant", False)]
    recall = Metrics.recall_at_k(retrieved_docs, relevant_docs, k=top_k)

    # 3. Cohérence sémantique sur les embeddings des docs récupérés
    embeddings = np.array([doc["embedding"] for doc in docs])
    coherence = Metrics.semantic_coherence(embeddings)

    # 4. Précision linguistique (comparaison simple)
    reference_texts = [doc["text"] for doc in docs if doc.get("is_relevant", False)]
    linguistic_precision = Metrics.linguistic_precision(generated_answer, reference_texts)

    return {
        "recall@k": recall,
        "coherence": coherence,
        "linguistic_precision": linguistic_precision,
        "generated_answer": generated_answer
    }
