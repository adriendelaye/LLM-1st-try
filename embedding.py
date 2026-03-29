    # embedding.py

    from sentence_transformers import SentenceTransformer
    import numpy as np
    import os
    import pickle

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    class Embedder:
        def __init__(self, model_name=MODEL_NAME, batch_size=32):
            self.model = SentenceTransformer(model_name)
            self.batch_size = batch_size
            self._cache = {}

        def embed_texts(self, texts, batch_size=None):
            if batch_size is None:
                batch_size = self.batch_size

            embeddings = []
            for text in texts:
                if text in self._cache:
                    embeddings.append(self._cache[text])
                else:
                    emb = self.model.encode(
                        text,
                        convert_to_numpy=True,
                        batch_size=batch_size
                    ).astype("float32")

                    self._cache[text] = emb
                    embeddings.append(emb)

            return np.array(embeddings)


    # Build embeddings
    def build_embeddings():
        emb_path = "models/embeddings.npy"
        sent_path = "models/sentences.pkl"

        if os.path.exists(emb_path) and os.path.exists(sent_path):
            print("Embeddings existing : skipping embeddings")
            return

        print("Construction des embeddings...")

        model = SentenceTransformer(MODEL_NAME)

        sentences = []

        for file in os.listdir("data/processed"):
            with open(f"data/processed/{file}", encoding="utf-8") as f:
                sentences += f.read().split("\n")

        sentences = [s.strip() for s in sentences if s.strip()]

        embeddings = model.encode(sentences, show_progress_bar=True)

        os.makedirs("models", exist_ok=True)

        np.save(emb_path, embeddings)

        with open(sent_path, "wb") as f:
            pickle.dump(sentences, f)

        print(f"{len(sentences)} encoded sentences")


    # load embeddings
    def load_embeddings():
        embeddings = np.load("models/embeddings.npy")
        with open("models/sentences.pkl", "rb") as f:
            sentences = pickle.load(f)
        return embeddings, sentences
