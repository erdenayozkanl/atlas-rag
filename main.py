import argparse
import json
import os
from pathlib import Path
from rag.core import RagIndex, read_document, answer

def main():
    parser = argparse.ArgumentParser(description='AtlasRAG: local hybrid retrieval and evaluation')
    subs = parser.add_subparsers(dest='command', required=True)
    ingest = subs.add_parser('ingest'); ingest.add_argument('path', nargs='?', default='data/knowledge.md')
    ask = subs.add_parser('ask'); ask.add_argument('question'); ask.add_argument('--k', type=int, default=3)
    subs.add_parser('eval'); subs.add_parser('serve')
    args = parser.parse_args()
    if args.command == 'serve':
        import uvicorn
        uvicorn.run('api:app', host='127.0.0.1', port=8001)
        return
    index = RagIndex()
    if args.command == 'ingest':
        path=Path(args.path)
        if path.is_dir():
            files=[p for p in path.rglob('*') if p.suffix.lower() in ('.md','.txt','.pdf')]
        else:
            files=[path]
        total=sum(index.ingest(str(f), read_document(f)) for f in files)
        print(f'Indexed {total} chunks from {len(files)} document(s) using {index.embedder.name}')
    if args.command == 'ask':
        print(json.dumps(answer(args.question,index.search(args.question,args.k),os.getenv('GENERATION_BACKEND','extractive')),indent=2))
    if args.command == 'eval':
        dataset=json.loads(Path('data/eval.json').read_text())
        hits=rr=0
        for item in dataset:
            results=index.search(item['question'],3)
            ranks=[j+1 for j,p in enumerate(results) if any(t.casefold() in p['section'].casefold() for t in item['relevant_terms'])]
            rank=min(ranks) if ranks else 0
            hits+=bool(rank); rr+=(1/rank if rank else 0)
            print(f"{'HIT' if rank else 'MISS'} rank={rank} {item['question']}")
        print(json.dumps({'recall@3':hits/len(dataset),'mrr@3':rr/len(dataset),'count':len(dataset)},indent=2))
if __name__=='__main__': main()
