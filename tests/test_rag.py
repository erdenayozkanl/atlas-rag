import json
from rag.core import RagIndex, chunk_text, HashEmbedder

def test_chunks_keep_heading_and_overlap():
    chunks=chunk_text('# Rules\n'+' '.join(f'word{i}' for i in range(22)), 'test',10,3)
    assert chunks[0]['section']=='Rules'
    assert 'word7' in chunks[0]['text'] and 'word7' in chunks[1]['text']

def test_ingest_idempotent_and_retrieval(tmp_path):
    index=RagIndex(str(tmp_path/'rag.db'),HashEmbedder())
    index.ingest('manual','# Webhooks\nVerify signatures using HMAC-SHA256.')
    index.ingest('manual','# Webhooks\nVerify signatures using HMAC-SHA256.')
    results=index.search('webhook signature verification')
    assert len(results)==1 and results[0]['section']=='Webhooks'
