# AtlasRAG — Local RAG Engineering Lab

Real chunking, reproducible embeddings, persistent SQLite index, cosine vector search, BM25, reciprocal rank fusion (RRF), sources/citations, CLI, REST API, and retrieval evaluation. Demo content belongs to fictional **NovaDesk**.

## Fast start (Python 3.11+)

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py ingest
python main.py ask "How do I verify webhooks?"
python main.py eval
python main.py serve
# POST http://127.0.0.1:8001/query with {"question":"What is the rate limit?"}
```

**Important:** The zero-download default is `hash-lexical-v1`, a deterministic lexical feature vector, **not** a neural semantic embedding. It exists for fast offline smoke tests.

Container API: `docker compose up --build` (port 8001). The SQLite index is mounted under `./data`; ingest documents before querying a fresh container.

## Actual semantic embeddings (recommended)

```bash
pip install -r requirements-semantic.txt
# Linux/macOS:
export EMBEDDING_BACKEND=semantic
# Windows PowerShell: $env:EMBEDDING_BACKEND="semantic"
python main.py ingest
python main.py eval
```

Uses `sentence-transformers/all-MiniLM-L6-v2`; first use downloads model weights (internet required). The model is English-oriented. Re-ingest whenever you change embedding models. Chunk size is measured in **words** here (not tokenizer tokens); configurable in `rag/core.py`.

## Optional local LLM answer generation

Install Ollama, run `ollama pull llama3.2`, start its server, then set `GENERATION_BACKEND=ollama`. Otherwise retrieval returns source excerpts and explicitly does **not** pretend to generate an LLM answer.

## Evaluation

`data/eval.json` contains an openly visible tiny sanity-check dataset. The `eval` command reports Recall@3 and MRR@3 based on expected document sections. These are **retrieval metrics**, not faithfulness or full answer quality. Extend with held-out realistic data, leakage protection, nDCG, rerankers, multilingual eval, latency p95, cost, LLM-as-judge with calibration, and prompt-injection tests before describing this as production-ready.

## Security and limitations

The demo server binds to localhost only; it has no authentication or per-tenant isolation. PDFs with images need separate OCR. Do not expose this API publicly without auth, quotas, audit logs, and document access controls.

```bash
pytest -q
```
