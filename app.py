import os
import time
import hashlib
import uuid
from typing import List, Optional, Dict, Any
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Body, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# ----------------------------
# Environment configuration
# ----------------------------

REQUIRED_ENV = [
    "GEMINI_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
]
for var in REQUIRED_ENV:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dq_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-1.5-flash")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")  # optional bearer token for relay

# ----------------------------
# Clients
# ----------------------------

genai.configure(api_key=GEMINI_API_KEY)

@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection(client: QdrantClient, dim: int = 768):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

# ----------------------------
# Schemas
# ----------------------------

class Metadata(BaseModel):
    source_type: Optional[str] = Field(None, description="e.g., profile_report, dq_rules, dq_run_report")
    source_name: Optional[str] = None
    path_or_table: Optional[str] = None
    timestamp: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

class UpsertItem(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[Metadata] = None

class UpsertBatchRequest(BaseModel):
    items: List[UpsertItem]
    chunk_size: int = 1200
    chunk_overlap: int = 150

class ChatRequest(BaseModel):
    query: str
    top_k: int = 6
    filters: Optional[Dict[str, Any]] = None  # e.g., {"source_type": "dq_rules"}
    max_output_tokens: int = 1024
    temperature: float = 0.2

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

	
# ----------------------------
# New Schemas for Recommendation
# ----------------------------

class RecommendationRequest(BaseModel):
    profile_summary: str = Field(..., description="Data profile summary text from Streamlit app")
    max_output_tokens: int = 1024
    temperature: float = 0.3

class RecommendationResponse(BaseModel):
    recommendations: str

# ----------------------------
# Auth dependency
# ----------------------------

def check_auth(authorization: Optional[str] = Header(None)):
    if AUTH_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.split(" ", 1)[1]
        if token != AUTH_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden")

# ----------------------------
# Utils
# ----------------------------

def stable_id(text: str, metadata: Optional[Metadata]) -> str:
    base = (text or "") + "|" + (metadata.source_type if metadata and metadata.source_type else "")
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens):
            break
        i += max(1, chunk_size - overlap)
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = genai.embed_content(model=EMBED_MODEL, content=texts, task_type="retrieval_document")
    # API returns {"embedding": [...]} for single input OR {"embeddings": [{"values": [...]}, ...]} for batch.
    if "embeddings" in model:
        return [e["values"] for e in model["embeddings"]]
    elif "embedding" in model:
        return [model["embedding"]]
    else:
        # Some client versions return .embeddings directly
        try:
            return [e.values for e in model.embeddings]
        except Exception:
            raise HTTPException(status_code=500, detail="Unexpected embedding response format")

def embed_query(text: str) -> List[float]:
    model = genai.embed_content(model=EMBED_MODEL, content=text, task_type="retrieval_query")
    if "embedding" in model:
        return model["embedding"]
    try:
        return model.embedding
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected embedding response format")

def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not filters:
        return None
    conditions = []
    for k, v in filters.items():
        conditions.append(FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v)))
    return Filter(should=conditions) if conditions else None

def cite_block(points: List[PointStruct]) -> str:
    lines = []
    for p in points:
        meta = p.payload.get("metadata", {})
        label = meta.get("source_type", "doc")
        name = meta.get("source_name", "")
        path = meta.get("path_or_table", "")
        lines.append(f"- [{label}] {name} {path}".strip())
    return "\n".join(lines)

# ----------------------------
# App
# ----------------------------

app = FastAPI(title="DQ Relay", version="1.0.0")

# CORS
origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    client = get_qdrant()
    # Assume text-embedding-004 has dim 768
    ensure_collection(client, dim=768)

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.post("/upsert_batch")
def upsert_batch(payload: UpsertBatchRequest, _: None = Depends(check_auth)):
    client = get_qdrant()
    points: List[PointStruct] = []
    for item in payload.items:
        md = item.metadata or Metadata()
        md.timestamp = md.timestamp or time.time()

        chunks = chunk_text(item.text, payload.chunk_size, payload.chunk_overlap)
        vectors = embed_texts(chunks)

        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = item.id or str(uuid.uuid4())
            print("Embedding vector shape", type(vec), len(vec), type(vec[0]))
            print("Sample Vector updated new", vec[:5])
            flat_vec=vec[0] if isinstance(vec,list) and isinstance(vec[0],list) else vec
            points.append(
                PointStruct(
                    id=pid,
                    vector={"default":flat_vec},
                    payload={
                        "text": chunk,
                        "metadata": md.dict(exclude_none=True),
                    },
                )
            )

    if not points:
        raise HTTPException(status_code=400, detail="No content to upsert")

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return {"upserted": len(points)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: None = Depends(check_auth)):
    client = get_qdrant()
    qvec = embed_query(req.query)

    q_filter = build_filter(req.filters)
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=("default", qvec),
        limit=max(1, req.top_k),
        with_payload=True,
        score_threshold=None,
        query_filter=q_filter,
    )

    # Build sources list
    sources = []
    for r in results:
        meta = r.payload.get("metadata", {})
        sources.append({
            "score": r.score,
            "source_type": meta.get("source_type"),
            "source_name": meta.get("source_name"),
            "path_or_table": meta.get("path_or_table"),
        })

    # Build prompt
    system_prompt = (
        "You are a data quality assistant. Answer using only the provided context. "
        "If the answer is not in context, say you do not have that information. "
        "Prefer precise references to rules, profile fields, and execution outcomes."
    )
    context_block = "\n\n---\n\n".join(
        f"[{meta.get('source_type','doc')}] {meta.get('source_name','')} {meta.get('path_or_table','')}\n{r.payload.get('text','')}"
        for r in results
    ) if results else "No context."
    user_prompt = f"{system_prompt}\n\nContext:\n{context_block}\n\nUser question:\n{req.query}\n\nWhen you cite or refer, mention the source_type or rule names if present."

    # Call Gemini
    model = genai.GenerativeModel(GEN_MODEL)
    resp = model.generate_content(
        user_prompt,
        generation_config={
            "max_output_tokens": req.max_output_tokens,
            "temperature": req.temperature,
        },
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ]
    )

    # Extract answer safely
    answer = ""
    if resp.candidates:
        cand = resp.candidates[0]
        if cand.finish_reason == "SAFETY":
            answer = "?? Response blocked by Gemini safety filters."
        elif cand.content.parts:
            answer = "".join(p.text for p in cand.content.parts if hasattr(p, "text"))

    if not answer:
        answer = "I don't have sufficient indexed context to answer that yet. Please load your reports or rules via /upsert_batch."

    return ChatResponse(answer=answer, sources=sources)


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest, _: None = Depends(check_auth)):
    """
    Generate Data Quality / Cleansing rule recommendations
    directly from Gemini based on a profile summary.
    """
    system_prompt = (
        "You are a Data Quality and Data Cleansing expert. "
        "Given a dataset profile summary, recommend specific DQ checks, "
        "validation rules, and cleansing strategies. "
        "Be precise, actionable, and structured."
    )

    user_prompt = f"{system_prompt}\n\nProfile Summary:\n{req.profile_summary}\n\nRecommendations:"

    model = genai.GenerativeModel(GEN_MODEL)
    resp = model.generate_content(
        user_prompt,
        generation_config={
            "max_output_tokens": req.max_output_tokens,
            "temperature": req.temperature,
        },
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ]
    )

    answer = ""
    if resp.candidates:
    cand = resp.candidates[0]
    if cand.finish_reason == "SAFETY":
        answer = "Response blocked by Gemini safety filters."
    else:
        # Try multiple ways to extract text
        if hasattr(cand, "content") and cand.content:
            # Case 1: parts with .text
            if hasattr(cand.content, "parts"):
                texts = [getattr(p, "text", "") for p in cand.content.parts]
                answer = "".join([t for t in texts if t])
            # Case 2: content is already a string
            elif isinstance(cand.content, str):
                answer = cand.content
            # Case 3: dict-like
            elif isinstance(cand.content, dict) and "parts" in cand.content:
                texts = [p.get("text", "") for p in cand.content["parts"]]
                answer = "".join([t for t in texts if t])
    # Fallback
    if not answer:
        answer = "Gemini returned no usable text. Please check profile summary formatting."