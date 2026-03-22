## LLM-1st-try

# My journey into coding from linguistics and mathematics background.

I learned HTML CSS for few classes that brang me almost 0 knowledge.
During my classes of linguistics, I became a tutor in informatic linguistics although I always felt fraudulent in comparison with my other tutorings.
Launching my GitHub, I am trying to display my intermediate skills in coding as I wish to become a computational linguist with linguistics as dominant.

# Advanced RAG System (Hybrid Retrieval + Reranking)

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline with modern best practices.

## Features

**Dense Retrieval** with FAISS (HNSW)
**Sparse Retrieval** with BM25
**Hybrid Search** using Reciprocal Rank Fusion (RRF)
**Neural Reranking** with CrossEncoder
**Prompt Engineering** for controlled LLM outputs
**Local LLM Generation** via Ollama (Mistral)

## Architecture

Query embedding (SentenceTransformers)
Hybrid retrieval (FAISS + BM25)
Reranking (CrossEncoder)
Context selection
Prompt construction
LLM generation

## Setup

```bash
pip install -r requirements.txt
