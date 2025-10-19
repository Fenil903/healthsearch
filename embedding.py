import os
import numpy as np
EMBEDDING_DIM = 384  # default dim for many SBERT models

# Try to load sentence-transformers if available
USE_REAL = False
try:
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    EMBEDDING_DIM = model.get_sentence_embedding_dimension()
    USE_REAL = True
except Exception as e:
    model = None
    # falls back to mock

def get_embedding(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        return None
    if USE_REAL and model is not None:
        emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(float)
    # Mock deterministic embedding: hash chunks -> vector
    arr = np.zeros(EMBEDDING_DIM, dtype=float)
    # simple deterministic approach: rolling char ords
    for i, ch in enumerate(text):
        idx = i % EMBEDDING_DIM
        arr[idx] += (ord(ch) % 97) + 1
    # normalize
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return (arr / norm).astype(float)
