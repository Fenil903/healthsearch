import sys
from pathlib import Path

# Add parent directory to path so we can import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from main import app
from embedding import get_embedding
import numpy as np
import os

# Setup static token for tests
os.environ["SVC_TOKEN"] = "super-secret-token"
client = TestClient(app)
HEADERS = {"Authorization": "Bearer super-secret-token"}

# -------------------- UNIT TESTS -------------------- #

def test_embedding_generation_is_deterministic():
    """Ensure embeddings are consistent and normalized."""
    text = "Patient has fever and cough."
    emb1 = get_embedding(text)
    emb2 = get_embedding(text)
    # Both embeddings should be identical for same input
    assert np.allclose(emb1, emb2)
    # Should be normalized (unit vector)
    norm = np.linalg.norm(emb1)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_add_and_search():
    """Verify note ingestion and semantic search."""
    # Add a note
    r = client.post(
        "/add_note",
        json={"patient_id": "P1", "note": "Patient has chest pain."},
        headers=HEADERS
    )
    assert r.status_code == 201
    data = r.json()
    assert data["patient_id"] == "P1"

    # Search for a related term
    s = client.get("/search_notes", params={"q": "chest"}, headers=HEADERS)
    assert s.status_code == 200
    result = s.json()
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "chest" in result[0]["note"].lower()


def test_missing_token_rejected():
    """Check that authentication is required."""
    r = client.post(
        "/add_note",
        json={"patient_id": "P2", "note": "No auth header test."}
    )
    assert r.status_code == 422 or r.status_code == 401


def test_empty_query_validation():
    """Ensure query validation catches empty search terms."""
    r = client.get("/search_notes", params={"q": ""}, headers=HEADERS)
    # FastAPI should return 422 Unprocessable Entity for missing/invalid query
    assert r.status_code == 422
