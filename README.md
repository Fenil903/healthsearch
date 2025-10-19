# HealthSearch - Semantic Medical Notes Search Service

A production-ready FastAPI microservice for semantic search over clinical notes using vector embeddings. Built for the Checkmed SDE technical assessment.

## 🩺 Overview

HealthSearch enables healthcare providers to store and intelligently search patient notes by meaning rather than keywords. The service generates vector embeddings for clinical notes and performs cosine-similarity-based retrieval to surface the most relevant notes, improving clinical decision-making and reducing search time.

**Core Capabilities:**
- Store clinical notes with automatically generated embeddings
- Search notes semantically (find "chest pain" when searching for "cardiac symptoms")
- Token-based authentication for API security
- Top-3 ranked results with similarity scores
- Persistent JSON-based storage with thread-safe operations

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (tested on Python 3.13)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Installation & Setup

#### Windows
```bash
# Clone repository
git clone https://github.com/Fenil903/healthsearch.git
cd healthsearch

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn main:app --reload
```

#### macOS/Linux
```bash
git clone https://github.com/Fenil903/healthsearch.git
cd healthsearch

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Configuration

Set the authentication token via environment variable:
```bash
# Default is "super-secret-token" if not set
export SVC_TOKEN="your-custom-token"
```

Optional: Specify the Sentence-Transformers model (defaults to `all-MiniLM-L6-v2`):
```bash
export SBERT_MODEL="all-mpnet-base-v2"  # Higher quality, slower
```

---

## 📡 API Endpoints

### 1. Add Note
**POST** `/add_note`

Store a new clinical note with automatic embedding generation.

**Request:**
```bash
curl -X POST "http://localhost:8000/add_note" \
  -H "Authorization: Bearer super-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "note": "Patient reports chest pain and shortness of breath. EKG normal."
  }'
```

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "patient_id": "P001",
  "note": "Patient reports chest pain and shortness of breath. EKG normal."
}
```

**Status Codes:**
- `201`: Note successfully added
- `400`: Empty note text or validation failure
- `401`: Missing or invalid authentication token
- `422`: Missing required fields or invalid format

### 2. Search Notes
**GET** `/search_notes`

Perform semantic search across stored notes.

**Request:**
```bash
curl -X GET "http://localhost:8000/search_notes?q=chest%20pain" \
  -H "Authorization: Bearer super-secret-token"
```

**Response (200 OK):**
```json
[
  {
    "patient_id": "P001",
    "note": "Patient reports chest pain and shortness of breath. EKG normal.",
    "score": 0.87
  },
  {
    "patient_id": "P003",
    "note": "Patient experiencing chest discomfort and palpitations.",
    "score": 0.82
  },
  {
    "patient_id": "P002",
    "note": "Mild chest wall tenderness on palpation.",
    "score": 0.71
  }
]
```

**Status Codes:**
- `200`: Search successful (returns 0-3 results)
- `400`: Empty search query
- `401`: Missing or invalid authentication token
- `422`: Missing query parameter or invalid format

### 3. Health Check
**GET** `/health`

Simple endpoint to verify service is running.

**Response:**
```json
{
  "status": "ok",
  "stored": 42
}
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test Categories

**Functionality Tests:**
- Note ingestion with proper embedding generation
- Semantic search accuracy and top-3 ranking
- Result sorting by similarity score

**Authentication Tests:**
- Token validation and rejection of invalid tokens
- Missing Authorization header handling
- Invalid Bearer format detection

**Validation Tests:**
- Missing required fields (patient_id, note, query)
- Empty input validation
- Malformed JSON handling

**Integration Tests:**
- Complete workflow: add note → search → verify results
- Edge cases (no matches, single result, multiple identical scores)

---

## 🏗️ Architecture & Design Decisions

### Project Structure
```
healthsearch/
├── main.py                 # FastAPI app & endpoints
├── auth.py                 # Token-based authentication
├── embedding.py            # Embedding generation (real/mock)
├── storage.py              # Persistent JSON storage & search logic
├── requirements.txt        # Dependencies
├── data.json              # Persistent note storage
├── README.md              # This file
└── tests/
    └── test_api.py        # Pytest unit & integration tests
```

### Technology Choices & Rationale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Framework** | FastAPI | Async, modern, auto-generated OpenAPI docs, built-in validation |
| **Embeddings** | Sentence-Transformers | Lightweight, accurate semantic representations; falls back to mock |
| **Storage** | JSON + Threading | Simple, human-readable, thread-safe for single-server; easy local testing |
| **Auth** | Bearer tokens | Stateless, simple to implement, sufficient for assessment scope |
| **Search** | Cosine similarity | Standard metric for normalized embeddings, mathematically sound |
| **Testing** | pytest + TestClient | Industry standard, good async support, comprehensive assertions |

### Architecture Patterns

**Clean Separation of Concerns:**
- `main.py`: HTTP request/response handling
- `auth.py`: Authentication logic (can be swapped for OAuth2)
- `embedding.py`: AI/ML abstraction (real or mock)
- `storage.py`: Data persistence and search algorithms

**Dependency Injection:**
- Token validation via `Depends(require_token)` decorator
- Enables easy mocking and testing

**Graceful Degradation:**
- Falls back to deterministic mock embeddings if Sentence-Transformers unavailable
- Ensures service runs in any environment

### Trade-offs & Design Decisions

**JSON Storage vs. PostgreSQL**
- **Chosen:** JSON file with thread-locking
- **Rationale:** Faster development, no external dependencies, sufficient for assessment
- **Trade-off:** Less scalable; for production, would migrate to PostgreSQL + pgvector

**Mock Embeddings Fallback**
- **Chosen:** Deterministic character-based embedding generation
- **Rationale:** Ensures tests run without ML library, demonstrates embedding interface
- **Trade-off:** Lower semantic quality; real Sentence-Transformers recommended for production

**In-Memory Search**
- **Chosen:** Load all embeddings into NumPy array, compute cosine similarity
- **Rationale:** Fast for small datasets (<10K notes), simple implementation
- **Trade-off:** O(n) per query; would use FAISS or Pinecone for millions of notes

**Bearer Token Authentication**
- **Chosen:** Simple static token validation
- **Rationale:** Sufficient for assessment, quick to implement
- **Trade-off:** No token refresh, user management, or role-based access

### Data Model

```python
NoteRecord = {
    "id": "uuid",                      # Unique identifier
    "patient_id": "string",            # Healthcare identifier
    "note": "string",                  # Clinical note text
    "embedding": [float, ...],         # Vector (384-dim for MiniLM)
    "score": float                     # Similarity score (search results only)
}
```

---

## 📊 Implementation Details

### Embedding Pipeline
1. **Input:** Raw clinical note text
2. **Normalization:** Strip whitespace
3. **Generation:** 
   - Production: Sentence-Transformers `all-MiniLM-L6-v2` (384 dimensions)
   - Fallback: Deterministic character-based hashing
4. **Normalization:** L2 normalization (unit vector)
5. **Storage:** Float32 array → JSON array

### Search Algorithm
1. **Query Embedding:** Generate embedding for search query (same pipeline)
2. **Similarity Computation:** Cosine similarity = dot product (since normalized)
3. **Ranking:** Sort by similarity descending
4. **Top-K Selection:** Return top 3 results with scores
5. **Response:** JSON with patient_id, note text, and similarity_score

### Thread Safety
- `Storage._lock` (threading.Lock) protects read/write operations
- JSON file operations are atomic within lock scope
- Suitable for single-server deployments

---

## 🔐 Security Considerations

### Implemented
✅ Bearer token authentication on all endpoints  
✅ Input validation (min_length constraints on text fields)  
✅ Proper HTTP status codes (401 for auth, 422 for validation)  
✅ Sanitized error messages (no internal stack traces exposed)  

### Recommendations for Production
- Implement OAuth2 with token expiration
- Add rate limiting (prevent brute force search)
- Encrypt sensitive data at rest (patient notes in encrypted storage)
- HTTPS-only communication
- Audit logging for compliance (HIPAA/GDPR)
- Database encryption and backup strategies

---

## 🎯 Project Metadata

### Time Investment
- **Estimated:** 2-4 hours (per requirements)
- **Actual:** ~3.5 hours
  - Core API implementation: 1 hour
  - Embedding & search logic: 45 minutes
  - Testing suite: 45 minutes
  - Documentation: 1 hour

### Bonus Features Implemented
✅ **Unit Testing** (pytest with comprehensive test coverage)  
- test cases covering happy paths, edge cases, and error conditions
- Deterministic mock embeddings enable consistent test results
- No external dependencies required for testing

❌ Docker Containerization (not implemented – prioritized core quality)  
❌ PostgreSQL Integration (would require pgvector extension setup)

### Known Limitations & Future Improvements

**Current Limitations:**
- JSON file storage not suitable for >50K notes (performance degrades)
- No pagination for search results (fixed top-3 limit)
- No note updates/deletion endpoints (append-only)
- Single-server only (no distributed caching)
- No semantic caching (same query re-computed each time)
- Mock embeddings have lower semantic quality than real models

**Recommended Production Enhancements:**
1. **Database Migration:** PostgreSQL + pgvector for native vector similarity
2. **Caching Layer:** Redis for frequently searched queries
3. **Async Search:** Batch processing for large note ingestions
4. **API Versioning:** Support `/v1/`, `/v2/` endpoints for breaking changes
5. **Enhanced Auth:** OAuth2 with user/client management
6. **Monitoring:** Prometheus metrics for search latency, embedding quality
7. **Note Management:** CRUD endpoints for updates/soft deletes
8. **Batch Operations:** `/bulk_add_notes` endpoint for bulk imports
9. **Similarity Threshold:** Make top-K configurable per query
10. **Embedding Quality Metrics:** Monitor avg similarity scores, outlier detection

---

## 🤔 Assumptions & Design Decisions

1. **Single Patient per Note:** Each note belongs to one patient_id (no multi-patient notes)
2. **Immutable Notes:** Notes cannot be updated after creation (by design)
3. **No User Authentication:** Token is global (not per-user); suitable for service-to-service auth
4. **Deterministic Search:** Same query always returns same results in same order
5. **English Language:** Embeddings optimized for English clinical text
6. **Local Development:** No cloud dependencies required
7. **Stateless API:** Each request is independent (no session state)
8. **Synchronous Embedding:** Embedding generation is blocking (acceptable for small notes)

---


## 📝 Example Workflows

### Workflow 1: Add Notes and Search
```bash
# Terminal 1: Start server
uvicorn main:app --reload

# Terminal 2: Add three notes
curl -X POST "http://localhost:8000/add_note" \
  -H "Authorization: Bearer super-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P001", "note": "Acute myocardial infarction. Elevated troponin."}'

curl -X POST "http://localhost:8000/add_note" \
  -H "Authorization: Bearer super-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P002", "note": "Chest pain, stable angina, no acute findings."}'

curl -X POST "http://localhost:8000/add_note" \
  -H "Authorization: Bearer super-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P003", "note": "Fractured rib from motor vehicle accident."}'

# Search for cardiac-related notes
curl -X GET "http://localhost:8000/search_notes?q=heart%20attack" \
  -H "Authorization: Bearer super-secret-token"

# Results will rank cardiac conditions highest, orthopedic injury lowest
```

### Workflow 2: Run Tests
```bash
pytest tests/test_api.py -v --tb=short
# Output shows 16 passed tests covering all endpoints and edge cases
```

---

## 🙏 Thank You

This implementation demonstrates practical application of modern backend technologies in healthcare context. Built with attention to code quality, security, and comprehensive testing — reflecting production engineering standards at Checkmed.

For questions or clarifications about the implementation, refer to the inline code comments or reach out to discuss architectural choices and trade-offs during the technical interview.

---

**Last Updated:** October 2025  
**Status:** Ready for Review