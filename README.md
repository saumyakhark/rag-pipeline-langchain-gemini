# RAG Pipeline with LangChain, FAISS & Gemini

A minimal, working implementation of a **Retrieval-Augmented Generation (RAG)** pipeline built in Google Colab. This project was built as part of the *Master RAG: Build an AI Retrieval System from Scratch* workshop hosted by GeeksforGeeks.

---

## What is RAG?

Large Language Models (LLMs) are powerful but limited to their training data. RAG extends them by allowing the model to **retrieve relevant context from your own documents** before generating an answer, making responses accurate, grounded, and specific to your data.

Instead of the AI guessing → it looks up first, then answers.

---

## Pipeline Overview

```
  Your Text/Document
       ↓
  Text Splitting (chunks)
       ↓
  Embeddings (HuggingFace)
       ↓
  Vector Store (FAISS)
       ↓
  Query → Retriever → Top-K Chunks
       ↓
  Prompt Template + Context
       ↓
  LLM (Gemini 2.5 Flash)
       ↓
  Grounded Answer
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Text Splitter | RecursiveCharacterTextSplitter |
| Environment | Google Colab |

---

## Project Structure

```
rag-pipeline-langchain-gemini/
│
├── rag_implementation.ipynb   # Main Colab notebook
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/rag-pipeline-langchain-gemini.git
```

### 2. Install dependencies
```bash
pip install langchain langchain-google-genai faiss-cpu langchain-text-splitters langchain-community
```

### 3. Set your Gemini API key

Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set it:
```python
os.environ["GEMINI_API_KEY"] = "your-api-key-here"
```

### 4. Add your knowledge base

Replace `ADD_KNOWLEDGE_BASE` in the notebook with your own text or document content.

### 5. Run the notebook

Open `rag_implementation.ipynb` in Google Colab and run all cells.

---

## Key Concepts Covered

- **Chunking** — Splitting large text into smaller overlapping segments (`chunk_size=200`, `chunk_overlap=50`)
- **Embeddings** — Converting text chunks into vector representations for semantic search
- **FAISS Vector Store** — Storing and retrieving vectors by similarity
- **Retriever** — Fetching top-K most relevant chunks for a given query (`k=3`)
- **Prompt Engineering** — Instructing the LLM to answer strictly from retrieved context
- **Gemini LLM** — Generating the final grounded answer

---

## Example Usage

```python
query = "What is the return policy?"
answer = rag_answer(query)
print("Question:", query)
print("Answer:", answer)
```

---
