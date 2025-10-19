from fastapi import FastAPI, HTTPException, Depends, Header, status, Query
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
from embedding import get_embedding, EMBEDDING_DIM
from auth import require_token
from storage import Storage

app = FastAPI(title="HealthSearch - Checkmed SDE Task")

DATA_FILE = Path("data.json")
storage = Storage(DATA_FILE)

class AddNoteRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    note: str = Field(..., min_length=1)

class SearchResult(BaseModel):
    patient_id: str
    note: str
    score: float

@app.post("/add_note", status_code=201)
async def add_note(payload: AddNoteRequest, token: str = Depends(require_token)):
    # generate embedding
    embedding = get_embedding(payload.note)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding")
    rec = storage.add_note(payload.patient_id, payload.note, embedding)
    return {"id": rec.id, "patient_id": rec.patient_id, "note": rec.note}

@app.get("/search_notes", response_model=List[SearchResult])
async def search_notes(q: str = Query(..., min_length=1), token: str = Depends(require_token)):
    q_emb = get_embedding(q)
    if q_emb is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding")
    results = storage.search(q_emb, top_k=3)
    return [SearchResult(patient_id=r.patient_id, note=r.note, score=r.score) for r in results]

@app.get("/health")
def health():
    return {"status": "ok", "stored": storage.count()}
