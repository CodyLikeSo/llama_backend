from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from faiss_answering import get_text

app = FastAPI()
# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get("/llama")
async def ask(response: str):
    return await get_text(query=response)