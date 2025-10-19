from typing import List, NamedTuple
import numpy as np
import json
from pathlib import Path
import threading
import uuid

class NoteRecord(NamedTuple):
    id: str
    patient_id: str
    note: str
    embedding: List[float]
    score: float = 0.0

class Storage:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._items = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    rec = NoteRecord(id=item["id"], patient_id=item["patient_id"], note=item["note"],
                                     embedding=item["embedding"], score=0.0)
                    self._items.append(rec)
            except Exception:
                # corrupt file -> start fresh
                self._items = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([{"id": r.id, "patient_id": r.patient_id, "note": r.note, "embedding": r.embedding} for r in self._items], f, ensure_ascii=False, indent=2)

    def add_note(self, patient_id: str, note: str, embedding: np.ndarray):
        with self._lock:
            rec_id = str(uuid.uuid4())
            emb_list = [float(x) for x in embedding.tolist()]
            rec = NoteRecord(id=rec_id, patient_id=patient_id, note=note, embedding=emb_list, score=0.0)
            self._items.append(rec)
            self._save()
            return rec

    def all_embeddings_matrix(self):
        if not self._items:
            return None
        mat = np.array([r.embedding for r in self._items], dtype=float)
        return mat

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        with self._lock:
            if not self._items:
                return []
            mat = self.all_embeddings_matrix()  # shape (n, dim)
            # ensure normalization
            q = np.array(query_embedding, dtype=float)
            # safe cosine sim: (q·x) / (||q||*||x||) but embeddings are normalized in our embedding generator
            # compute dot products
            dots = mat.dot(q)
            # compute norms if needed (but our get_embedding normalizes; still safe)
            mat_norms = np.linalg.norm(mat, axis=1)
            q_norm = np.linalg.norm(q)
            denom = mat_norms * (q_norm if q_norm != 0 else 1.0)
            sim = dots / np.where(denom == 0, 1.0, denom)
            # top k indices
            idx = np.argsort(-sim)[:top_k]
            results = []
            for i in idx:
                results.append(NoteRecord(id=self._items[i].id, patient_id=self._items[i].patient_id,
                                          note=self._items[i].note, embedding=self._items[i].embedding,
                                          score=float(sim[i])))
            return results

    def count(self):
        return len(self._items)
