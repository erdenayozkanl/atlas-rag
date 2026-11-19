"""Local RAG: heading-aware chunking, embedders, persistent index, hybrid retrieval."""
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())

def chunk_text(text: str, source: str, chunk_tokens: int = 110, overlap: int = 22) -> list[dict]:
    """Heading-aware word-window chunker with bounded overlap and provenance."""
    if not 0 <= overlap < chunk_tokens:
        raise ValueError('overlap must be smaller than chunk_tokens')
    heading = 'Untitled'
    sections = []
    buf = []
    def flush():
        if buf:
            sections.append((heading, ' '.join(buf)))
            buf.clear()
    for line in text.splitlines():
        if line.lstrip().startswith('#'):
            flush()
            heading = line.lstrip('# ').strip() or heading
        elif line.strip():
            buf.append(line.strip())
    flush()
    result = []
    for title, body in sections:
        words = body.split()
        step = chunk_tokens - overlap
        for start in range(0, len(words), step):
            part = words[start:start + chunk_tokens]
            if not part:
                break
            content = f'{title}\n' + ' '.join(part)
            digest = hashlib.sha256(f'{source}:{title}:{start}:{content}'.encode()).hexdigest()[:20]
            result.append({'id': digest, 'source': source, 'section': title,
                           'start_word': start, 'text': content})
            if start + chunk_tokens >= len(words):
                break
    return result

class HashEmbedder:
    """Offline lexical baseline, NOT a semantic neural embedding model."""
    name = 'hash-lexical-v1'
    dimension = 512
    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for term in tokens(text):
                digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], 'big') % self.dimension
                sign = 1 if digest[4] % 2 == 0 else -1
                matrix[i, index] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)

class SemanticEmbedder:
    """Real neural sentence embeddings; downloads model on first use."""
    name = 'sentence-transformers/all-MiniLM-L6-v2'
    dimension = 384
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('pip install -r requirements-semantic.txt') from exc
        self.model = SentenceTransformer(self.name)
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

def get_embedder():
    mode = os.getenv('EMBEDDING_BACKEND', 'hash').lower()
    if mode == 'hash':
        return HashEmbedder()
    if mode == 'semantic':
        return SemanticEmbedder()
    raise ValueError('EMBEDDING_BACKEND must be hash or semantic')

class RagIndex:
    def __init__(self, path: str = 'data/index.db', embedder=None):
        self.path = path
        self.embedder = embedder or get_embedder()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, source TEXT, section TEXT, text TEXT, vec TEXT, backend TEXT)')
    def ingest(self, source: str, text: str, chunk_tokens: int = 110, overlap: int = 22) -> int:
        chunks = chunk_text(text, source, chunk_tokens, overlap)
        if not chunks:
            return 0
        vectors = self.embedder.encode([c['text'] for c in chunks])
        with sqlite3.connect(self.path) as conn:
            conn.execute('DELETE FROM chunks WHERE source=?', (source,))
            conn.executemany('INSERT INTO chunks VALUES (?,?,?,?,?,?)', [
                (c['id'], source, c['section'], c['text'], json.dumps(v.tolist()), self.embedder.name)
                for c, v in zip(chunks, vectors)
            ])
        return len(chunks)
    def search(self, query: str, k: int = 3, candidate_k: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute('SELECT id,source,section,text,vec,backend FROM chunks').fetchall()
        if not rows:
            return []
        if any(r[5] != self.embedder.name for r in rows):
            raise ValueError('Embedding backend changed: reindex with ingest before searching')
        docs = [r[3] for r in rows]
        qvec = self.embedder.encode([query])[0]
        vectors = np.array([json.loads(r[4]) for r in rows], dtype=np.float32)
        similarities = vectors @ qvec
        # Lightweight BM25, with reciprocal-rank fusion between lexical and vector scores.
        corpus = [tokens(d) for d in docs]
        df = Counter(t for doc in corpus for t in set(doc))
        avgdl = sum(map(len, corpus)) / max(len(corpus), 1)
        bm25 = []
        for doc in corpus:
            tf = Counter(doc)
            score = 0.0
            for term in set(tokens(query)):
                idf = math.log(1 + (len(corpus) - df[term] + .5)/(df[term] + .5))
                score += idf * tf[term] * 2.2 / (tf[term] + 1.2 * (.25 + .75 * len(doc) / max(avgdl, 1)))
            bm25.append(score)
        vec_order = np.argsort(-similarities)
        lex_order = np.argsort(-np.asarray(bm25))
        fused = np.zeros(len(rows))
        for order in (vec_order, lex_order):
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (60 + rank + 1)
        # Transparent offline reranker; replace this seam with a cross-encoder in production.
        query_terms = set(tokens(query))
        candidates = np.argsort(-fused)[:max(k, candidate_k)]
        order = sorted(candidates, key=lambda i: (len(query_terms & set(corpus[i])), float(fused[i])), reverse=True)[:max(1, k)]
        return [{'id':rows[i][0], 'source':rows[i][1], 'section':rows[i][2],
                 'text': rows[i][3], 'score': round(float(fused[i]), 6), 'reranker':'term-coverage-v1'} for i in order]

    def evaluate(self, cases: list[dict], k: int = 3) -> dict[str, float]:
        hits = reciprocal = 0.0
        for case in cases:
            rank = next((i for i, item in enumerate(self.search(case['question'], k=k), 1) if item['section'] == case['expected_section']), None)
            hits += float(rank is not None); reciprocal += 1.0 / rank if rank else 0.0
        total = max(len(cases), 1)
        return {f'recall_at_{k}': hits / total, f'mrr_at_{k}': reciprocal / total, 'cases': len(cases)}

def read_document(path: Path) -> str:
    if path.suffix.lower() == '.pdf':
        from pypdf import PdfReader
        return '\n'.join(page.extract_text() or '' for page in PdfReader(str(path)).pages)
    if path.suffix.lower() in ('.md', '.txt'):
        return path.read_text(encoding='utf-8')
    raise ValueError('Only .txt, .md, .pdf are supported')

def answer(query: str, passages: list[dict], provider: str = 'extractive') -> dict:
    if not passages:
        return {'answer':'No indexed documents. Run ingest first.', 'citations':[], 'mode':provider}
    if provider == 'extractive':
        return {'answer':'Relevant source excerpts (not LLM-generated):\n' + '\n'.join(
            f"[{i}] {p['text']}" for i,p in enumerate(passages,1)),
            'citations':[{'source':p['source'], 'section':p['section']} for p in passages], 'mode':'extractive'}
    if provider == 'ollama':
        import urllib.request
        context = '\n\n'.join(f"[{i}] {p['text']}" for i,p in enumerate(passages,1))
        payload = json.dumps({'model':os.getenv('OLLAMA_MODEL','llama3.2'), 'stream':False,
                              'prompt':f'Answer ONLY from the source passages; if unknown say unknown. Cite passage numbers.\n{context}\nQuestion: {query}'}).encode()
        req = urllib.request.Request(os.getenv('OLLAMA_URL','http://localhost:11434') + '/api/generate', data=payload, headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.load(resp)
        return {'answer':data['response'],'citations':[{'source':p['source'],'section':p['section']} for p in passages], 'mode':'ollama'}
    raise ValueError('provider must be extractive or ollama')
