from fastapi import FastAPI
from pydantic import BaseModel
from rag.core import RagIndex, answer
import os
app = FastAPI(title='AtlasRAG', version='0.1.0')
class Question(BaseModel):
    question: str
    top_k: int = 3
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/query')
def query(request: Question):
    index=RagIndex()
    return answer(request.question,index.search(request.question, min(max(request.top_k,1),20)),os.getenv('GENERATION_BACKEND','extractive'))
