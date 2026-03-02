# CCPA Compliance System — MarkovDevs

> **OpenHack 2026 · CSA, IISc** — A high-precision Graph-RAG system designed to bridge the gap between natural language business practices and complex statutory law.

---

## 🚀 Why MarkovDevs Stands Out

Most RAG systems treat legal text as flat, disconnected chunks. Our approach recognizes that **law is a network of relationships**.

- **Graph-Augmented Retrieval:** Instead of simple semantic search, our system uses a Knowledge Graph. When a section is retrieved, the engine automatically follows `exemptions_in` and `modifies` edges. This ensures that if a prompt involves HIPAA-protected data, the system doesn't just see "data deletion"—it instantly pulls in the Section 1798.146 shield.

- **The "3B Density" Advantage:** We use a Q4_K_M quantization of Qwen-2.5-3B. While the hackathon allows 8B models, our 3B model is *statute-dense* — by providing pre-filtered, graph-expanded context, it achieves higher reasoning accuracy with near-instant inference on standard hardware.

- **Post-Inference Legal Verification:** Our pipeline includes a custom `LegalVerifier`. This layer cross-references LLM citations against our graph schema to strip out administrative sections or expired provisions, ensuring the final JSON contains only actionable legal articles.

---

## 🛠 Solution Overview

### Approach

We built a **Graph-RAG pipeline** on top of a CPU-quantized LLM. The CCPA statute is represented as a typed knowledge graph (52+ sections with specific edge types). This allows the retrieval system to surface not only the directly relevant sections but also their associated legal shields.

### Pipeline (Input → Output)

```
POST /analyze {"prompt": "..."}
        │
        ▼
  [1] FAISS Semantic Search (mxbai-embed-large-v1)
      Top-4 most semantically relevant CCPA sections
        │
        ▼
  [2] Graph Expansion
      Follow exemptions_in + mentions edges from each hit
      → pulls in applicable exemptions automatically
        │
        ▼
  [3] Qwen-2.5-3B-Instruct (GGUF Q4_K_M, CPU/GPU)
      Violation-first prompting + 5 typed few-shot examples
      (selling, undisclosed collection, deletion, pricing,
       minors, warranty exemption, HIPAA)
        │
        ▼
  [4] LegalVerifier
      • Validates citations against live graph schema
      • Strips Exemption / Administrative / Expired sections
      • Corrects contradictory harmful=true / articles=[] states
        │
        ▼
  {"harmful": bool, "articles": ["Section 1798.xxx", ...]}
```

### Models & Libraries

| Component | Details |
|---|---|
| **LLM** | `Qwen/Qwen2.5-3B-Instruct-GGUF` · Q4_K_M · ~2.5 GB |
| **Embeddings** | `mixedbread-ai/mxbai-embed-large-v1` · 512-dim |
| **Vector Index** | `faiss-cpu` — millisecond ANN search |
| **Knowledge Base** | Custom CCPA knowledge graph (JSON) with typed edges |
| **API Framework** | FastAPI + uvicorn |
| **Package Manager** | `uv` — 10–100× faster than pip |

Both models are **public on Hugging Face** — no HF token is required.

---

## 📦 Docker Run Command

> The container **pre-downloads all models at build time**. No internet access is required at runtime.

```bash
# With GPU (Recommended — ~5× faster inference)
docker run --gpus all -p 8000:8000 sahilrane/ccpa-compliance:latest

# CPU Only (Fully supported — Verified 10/10 PASS on CPU)
docker run -p 8000:8000 sahilrane/ccpa-compliance:latest
```

Using Docker Compose (organizer recommended flow):

```bash
docker compose up -d       # Start container
python test.py             # Run evaluation
docker compose down        # Cleanup
```

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | **No** | Hugging Face access token. Both models (`Qwen2.5-3B-GGUF` and `mxbai-embed-large-v1`) are **public** — token is not required. Only supply if you swap to a gated model. |

---

## 🖥️ GPU Requirements

| | Details |
|---|---|
| **Recommended GPU** | NVIDIA with ≥ 4 GB VRAM |
| **Minimum GPU** | 4 GB VRAM (model is Q4-quantized to ~2.5 GB) |
| **RAM** | 8 GB system RAM |
| **CPU Fallback** | ✅ Fully supported — `llama-cpp-python` auto-detects hardware |
| **Python** | 3.12 · uses `uv` for dependency management |

The `--gpus all` flag is safe to include even on CPU-only machines — the container starts normally and falls back to CPU inference automatically.

---

## 🐧 Local Setup Instructions (Fallback — Linux VM)

> Use this only if the Docker container fails to start. Note: manual deployment incurs a score penalty per organizer rules.

Requires: **Python 3.12**, install `uv` with `pip install uv`

```bash
# 1. Enter project directory
cd ccpa-compliance-system

# 2. Install all dependencies
uv sync

# 3. Build the CCPA knowledge graph (skip if data/ccpa_graph.json already exists)
uv run python scripts/02_build_graph.py

# 4. Build the FAISS vector index (skip if data/faiss_index/ already exists)
uv run python scripts/03_build_vector_db.py

# 5. Download models (~2.5 GB, first run only)
uv run python scripts/04_download_model.py

# 6. Start the FastAPI server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server is ready when you see: `✅ Engine Fully Loaded on CPU!`

---

## 📍 API Usage Examples

### GET /health — Server readiness

```bash
curl http://localhost:8000/health
```
```json
{"status": "ready"}
```

### POST /analyze — CCPA Violation detected

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We sell customer browsing history to ad networks without notifying them."}'
```
```json
{"harmful": true, "articles": ["Section 1798.120"]}
```

### POST /analyze — No violation (compliant practice)

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We provide a clear privacy policy and honor all deletion requests within 45 days."}'
```
```json
{"harmful": false, "articles": []}
```

### POST /analyze — Exemption applies (HIPAA)

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A patient requested deletion of their hospital records. We refused because they are governed by HIPAA."}'
```
```json
{"harmful": false, "articles": []}
```

---

## 🗂 Project Structure

```
ccpa-compliance-system/
├── app/
│   ├── main.py          # FastAPI app — /health + /analyze endpoints
│   ├── engine.py        # Graph-RAG engine (FAISS + graph expansion + LLM)
│   ├── verifier.py      # LegalVerifier — citation validation & type safety
│   └── schemas.py       # Pydantic request/response models
├── data/
│   ├── ccpa_graph.json        # CCPA knowledge graph (52 sections, typed edges)
│   └── faiss_index/
│       ├── ccpa.index         # FAISS ANN index
│       └── mapping.json       # Index position → section ID
├── scripts/
│   ├── 02_build_graph.py      # Build knowledge graph from statute data
│   ├── 03_build_vector_db.py  # Build FAISS index from graph
│   └── 04_download_model.py   # Pre-download models (runs at Docker build time)
├── Dockerfile                 # Two-stage build — models baked in at build time
├── docker-compose.yml
├── pyproject.toml             # Dependencies managed by uv
├── test.py                    # Official organizer test script
└── validate_format.py         # Format validation script
```

---

## 🧪 Test Results (Official `test.py`)

Our system achieves a **100% pass rate** on the official organizer test script.

| # | Scenario | Expected | Status |
|---|---|---|---|
| 1 | Selling data without opt-out | `harmful: true` | ✅ PASS |
| 2 | Undisclosed data collection | `harmful: true` | ✅ PASS |
| 3 | Ignoring deletion request | `harmful: true` | ✅ PASS |
| 4 | Discriminatory pricing for opt-out | `harmful: true` | ✅ PASS |
| 5 | Minor's data without consent | `harmful: true` | ✅ PASS |
| 6 | CCPA-compliant data practices | `harmful: false` | ✅ PASS |
| 7 | Proper deletion compliance | `harmful: false` | ✅ PASS |
| 8 | Unrelated request (team meeting) | `harmful: false` | ✅ PASS |
| 9 | Proper opt-out link | `harmful: false` | ✅ PASS |
| 10 | Non-discriminatory pricing | `harmful: false` | ✅ PASS |

**Final Score: 10/10 — Exit code 0** 🎉
