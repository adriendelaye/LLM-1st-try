# app.py

from pipeline import AdvancedRAGPipeline
from huggingface_hub import login

login("") #add your HuggingFace token in the strings
# Initialization of pipeline
pipeline = AdvancedRAGPipeline()

# Exemple of documents (OBLIGATORY for test)
docs = [
    {
        "id": "1",
        "text": "Le chat dort sur le tapis.",
        "concept": "chat",
        "type": "sentence",
        "domain": "pragmatics"
    },
    {
        "id": "2",
        "text": "Le chien aboie fort.",
        "concept": "chien",
        "type": "sentence",
        "domain": "pragmatics"
    }
]

# build index
texts = [doc["text"] for doc in docs]
embeddings = pipeline.embedder.embed_texts(texts)
pipeline.retriever.build_index(embeddings, docs)

# interactive loop
while True:
    query = input("\nYour question (or 'exit') : ")
    if query.lower() == "exit":
        break

    answer = pipeline(query)
    print("\nAnswer :\n", answer)
