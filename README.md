# HealthSearch (Checkmed SDE Task)

## 🩺 Overview
HealthSearch is a FastAPI-based microservice that stores and semantically searches patient notes using vector embeddings.

This project demonstrates clean backend design, authentication, vector-based retrieval, and robust unit testing — essential for reliability in healthcare software.

---

## ⚙️ Features
- **FastAPI** backend
- **Add & search notes** by meaning (semantic search)
- **Embedding generation** using Sentence-Transformers (or deterministic fallback)
- **Token-based authentication**
- **Persistent JSON storage**
- **Pytest unit tests for reliability**

---

## 🖥️ Setup (Windows)
```bash
git clone <your_repo_link>
cd healthsearch
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
