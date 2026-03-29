# prompting.py
def build_context(docs, max_docs=5):
    selected = docs[:max_docs]

    chunks = []
    for i, doc in enumerate(selected, 1):
        text = doc.get("text", "").strip()
        if not text:
            continue

        chunk = f"""[Document {i}]
Type: {doc.get('type', 'unknown')}
Concept: {doc.get('concept', 'unknown')}
Content: {text}
"""
        chunks.append(chunk)

    return "\n".join(chunks)

def build_prompt(query, context):
    return f"""You are an expert in linguistics and philosophy of language.

Answer the question using ONLY the provided context.

If the answer is not in the context, say: "I don't know."

---

CONTEXT:
{context}

---

QUESTION:
{query}

---

ANSWER (structured):
1. Definition
2. Explanation
3. Example
4. Summary
"""
