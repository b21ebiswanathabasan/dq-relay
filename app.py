import os
import time
import hashlib
import uuid
import json
from typing import List, Optional, Dict, Any, Tuple
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

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

PROJ_ROOT = os.getenv("PROJ_ROOT", ".")  # root for saving dw/rule/*.json

# Ensure dw/rule directory exists
DW_RULE_DIR = os.path.join(PROJ_ROOT, "dw", "rule")
os.makedirs(DW_RULE_DIR, exist_ok=True)

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
# Schemas (existing)
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
# New Schemas for Recommendation (existing)
# ----------------------------

class RecommendationRequest(BaseModel):
    profile_summary: str = Field(..., description="Data profile summary text from Streamlit app")
    max_output_tokens: int = 1024
    temperature: float = 0.3

class RecommendationResponse(BaseModel):
    recommendations: str

# ----------------------------
# New Schemas for Analytics (existing)
# ----------------------------

class AnalyticsRequest(BaseModel):
    query: str = Field(..., description="Analytics question, e.g. 'Top 10 failed dq rules for last 3 months'")
    top_k: int = 20
    filters: Optional[Dict[str, Any]] = None
    max_output_tokens: int = 1024
    temperature: float = 0.2

class AnalyticsResponse(BaseModel):
    analysis: str
    sources: List[Dict[str, Any]]

# ----------------------------
# New Schemas for NLP Rule Creation
# ----------------------------

ALLOWED_TYPES = ['not_null','regex','domain','range','unique','cross_field','freshness','referential']

class SchemaColumn(BaseModel):
    name: str
    dtype: str

class SchemaInput(BaseModel):
    dataset_alias: str
    path_or_table: str
    columns: List[SchemaColumn]

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

class Predicate(BaseModel):
    type: str
    expr: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

    @validator('type')
    def check_type(cls, v):
        if v not in ALLOWED_TYPES:
            raise ValueError(f"Unsupported condition.type '{v}'. Allowed: {ALLOWED_TYPES}")
        return v

class RuleTarget(BaseModel):
    dataset_alias: str
    path_or_table: str
    column: Optional[str] = None

class RuleModel(BaseModel):
    id: str
    name: str
    target: RuleTarget
    predicate: Predicate
    severity: str = Field(..., description="info | warn | error")
    active: bool = False
    warnings: Optional[List[str]] = None
    created_at: Optional[str] = None
    source_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @validator('severity')
    def check_severity(cls, v):
        if v not in ['info', 'warn', 'error']:
            raise ValueError("severity must be one of: info, warn, error")
        return v

class NlpRuleCreateRequest(BaseModel):
    text: str
    schema: SchemaInput
    auto_commit: Optional[bool] = True

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
    if "embeddings" in model:
        return [e["values"] for e in model["embeddings"]]
    elif "embedding" in model:
        return [model["embedding"]]
    else:
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

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def rule_to_text(rule: RuleModel) -> str:
    # Human-readable summary used for indexing
    pt = rule.predicate.type
    expr = rule.predicate.expr or ""
    params = rule.predicate.params or {}
    target = f"{rule.target.dataset_alias} {rule.target.path_or_table}" + (f" column {rule.target.column}" if rule.target.column else "")
    lines = [
        f"Rule: {rule.name}",
        f"Target: {target}",
        f"Predicate: {pt}",
        f"Expr: {expr}" if expr else "",
        f"Params: {json.dumps(params, ensure_ascii=False)}" if params else "",
        f"Severity: {rule.severity}",
        f"Active: {rule.active}",
    ]
    return "\n".join([l for l in lines if l])

def gen_rule_id() -> str:
    return "rule_" + uuid.uuid4().hex[:12]

def sanitize_filename(name: str) -> str:
    return "".join([c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in name])[:80]

# ----------------------------
# NLP parsing
# ----------------------------

def parse_rules_with_gemini(req: NlpRuleCreateRequest) -> List[RuleModel]:
    """
    Use Gemini to parse req.text into structured rules constrained to ALLOWED_TYPES.
    """
    schema_cols = ", ".join([f"{c.name}:{c.dtype}" for c in req.schema.columns])
    prompt = (
        "You are a Data Quality rule generator. Convert the user's natural language into a JSON array of rules. "
        "Each rule must have: name, target {dataset_alias, path_or_table, optional column}, "
        f"predicate {{ type one of {ALLOWED_TYPES}, optional expr, optional params }}, severity (info|warn|error). "
        "Do not invent columns that are not in the provided schema. If a condition mentions a column not present, include a warning in the rule. "
        "Return pure JSON, no markdown, no commentary.\n\n"
        f"Schema:\n- dataset_alias: {req.schema.dataset_alias}\n- path_or_table: {req.schema.path_or_table}\n"
        f"- columns: [{schema_cols}]\n\n"
        f"User text:\n{req.text}\n\n"
        "Rules JSON:"
    )

    model = genai.GenerativeModel(GEN_MODEL)
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 1024, "temperature": 0.1},
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ]
    )

    text = ""
    if hasattr(resp, "text") and resp.text:
        text = resp.text
    elif resp.candidates and resp.candidates[0].content and hasattr(resp.candidates[0].content, "parts"):
        text = "".join([getattr(p, "text", "") for p in resp.candidates[0].content.parts])

    if not text:
        raise HTTPException(status_code=500, detail="Model returned no rules")

    try:
        # Some models might wrap in code fences; strip if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # remove possible "json" tag
            cleaned = cleaned.replace("json", "", 1).strip()
        raw_rules = json.loads(cleaned)
        if not isinstance(raw_rules, list):
            raise ValueError("Expected a JSON array of rules")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse rules JSON: {e}")

    rules: List[RuleModel] = []
    for r in raw_rules:
        try:
            # Fill in target dataset from schema; enforce alias/table
            target = r.get("target", {}) or {}
            target.setdefault("dataset_alias", req.schema.dataset_alias)
            target.setdefault("path_or_table", req.schema.path_or_table)

            # Generate id, timestamps, metadata
            rid = gen_rule_id()
            created_at = now_iso()
            warnings = r.get("warnings", []) or []

            # Validate column presence
            col = target.get("column")
            if col and col not in req.schema.column_names:
                warnings.append(f"Column '{col}' not found in schema; please fix.")

            # Validate type
            pred = r.get("predicate", {})
            ptype = pred.get("type")
            if ptype not in ALLOWED_TYPES:
                warnings.append(f"Unsupported type '{ptype}'. Allowed: {ALLOWED_TYPES}")

            # Construct RuleModel (pydantic will enforce severity/type)
            rule = RuleModel(
                id=rid,
                name=r.get("name") or f"{ptype or 'rule'}_{col or 'dataset'}",
                target=RuleTarget(
                    dataset_alias=target["dataset_alias"],
                    path_or_table=target["path_or_table"],
                    column=col
                ),
                predicate=Predicate(
                    type=ptype,
                    expr=pred.get("expr"),
                    params=pred.get("params")
                ),
                severity=r.get("severity", "warn"),
                active=False,
                warnings=warnings,
                created_at=created_at,
                metadata={"source_type": "dw_rules"}
            )
            rules.append(rule)
        except Exception as e:
            # Skip unparseable rule, but continue
            raise HTTPException(status_code=400, detail=f"Rule validation error: {e}")

    return rules

def save_rule_to_disk(rule: RuleModel) -> str:
    fname = sanitize_filename(rule.name or rule.id) + ".json"
    fpath = os.path.join(DW_RULE_DIR, fname)
    payload = rule.dict()
    payload["metadata"] = {"source_type": "dw_rules", "path_or_table": rule.target.path_or_table, "source_name": rule.name}
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fpath

def index_rule_in_qdrant(rule: RuleModel):
    client = get_qdrant()
    text = rule_to_text(rule)
    vectors = embed_texts([text])
    vec = vectors[0]
    pid = rule.id
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=pid,
                vector={"default": vec},
                payload={
                    "text": text,
                    "metadata": {
                        "source_type": "dw_rules",
                        "source_name": rule.name,
                        "path_or_table": rule.target.path_or_table,
                        "timestamp": time.time(),
                        "extra": {
                            "id": rule.id,
                            "severity": rule.severity,
                            "column": rule.target.column,
                            "predicate": rule.predicate.type,
                        }
                    }
                }
            )
        ]
    )

# ----------------------------
# App
# ----------------------------

app = FastAPI(title="DQ Relay", version="1.1.0")

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
            flat_vec = vec[0] if isinstance(vec, list) and isinstance(vec[0], list) else vec
            points.append(
                PointStruct(
                    id=pid,
                    vector={"default": flat_vec},
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

    sources = []
    for r in results:
        meta = r.payload.get("metadata", {})
        sources.append({
            "score": r.score,
            "source_type": meta.get("source_type"),
            "source_name": meta.get("source_name"),
            "path_or_table": meta.get("path_or_table"),
        })

    system_prompt = (
        "You are a data quality assistant. Answer using only the provided context. "
        "If the answer is not in context, say you do not have that information. "
        "Prefer precise references to rules, profile fields, and execution outcomes."
    )
    context_block = "\n\n---\n\n".join(
        f"[{r.payload.get('metadata', {}).get('source_type','doc')}] {r.payload.get('metadata', {}).get('source_name','')} {r.payload.get('metadata', {}).get('path_or_table','')}\n{r.payload.get('text','')}"
        for r in results
    ) if results else "No context."
    user_prompt = f"{system_prompt}\n\nContext:\n{context_block}\n\nUser question:\n{req.query}\n\nWhen you cite or refer, mention the source_type or rule names if present."

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
            answer = "?? Response blocked by Gemini safety filters."
        elif getattr(cand, "content", None) and getattr(cand.content, "parts", None):
            answer = "".join(p.text for p in cand.content.parts if hasattr(p, "text"))

    if not answer:
        answer = "I don't have sufficient indexed context to answer that yet. Please load your reports or rules via /upsert_batch."

    return ChatResponse(answer=answer, sources=sources)

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest, _: None = Depends(check_auth)):
    system_prompt = (
        "You are a Data Quality and Data Cleansing expert. "
        "Given a dataset profile summary, recommend specific DQ checks, "
        "validation rules, and cleansing strategies. "
        "Be precise, actionable, and structured."
    )

    user_prompt = f"{system_prompt}\n\nProfile Summary:\n{req.profile_summary}\n\nRecommendations:"

    model = genai.GenerativeModel("gemini-2.5-flash")
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
    try:
        if hasattr(resp, "text") and resp.text:
            answer = resp.text
        elif resp.candidates:
            cand = resp.candidates[0]
            if cand.finish_reason == "SAFETY":
                answer = "Response blocked by Gemini safety filters."
            elif hasattr(cand, "content") and cand.content:
                if hasattr(cand.content, "parts"):
                    texts = [getattr(p, "text", "") for p in cand.content.parts]
                    answer = "".join([t for t in texts if t])
                elif isinstance(cand.content, str):
                    answer = cand.content
                elif isinstance(cand.content, dict) and "parts" in cand.content:
                    texts = [p.get("text", "") for p in cand.content["parts"]]
                    answer = "".join([t for t in texts if t])
    except Exception as e:
        print("Error extracting Gemini response:", e)

    if not answer:
        answer = "Gemini returned no usable text. Please check profile summary formatting."

    return RecommendationResponse(recommendations=answer)

@app.post("/analytics", response_model=AnalyticsResponse)
def analytics(req: AnalyticsRequest, _: None = Depends(check_auth)):
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

    sources = []
    for r in results:
        meta = r.payload.get("metadata", {})
        sources.append({
            "score": r.score,
            "source_type": meta.get("source_type"),
            "source_name": meta.get("source_name"),
            "path_or_table": meta.get("path_or_table"),
        })

    system_prompt = (
        "You are a Data Quality analytics assistant. "
        "Given context from data quality rules, profile reports, and cleansing runs, "
        "generate clear analytics and summaries. "
        "Focus on aggregations, trends, and top-N style answers (e.g., top 10 failed rules). "
        "If the answer is not in the context, say you do not have that information."
    )
    context_block = "\n\n---\n\n".join(
        f"[{r.payload.get('metadata', {}).get('source_type','doc')}] {r.payload.get('metadata', {}).get('source_name','')} {r.payload.get('metadata', {}).get('path_or_table','')}\n{r.payload.get('text','')}"
        for r in results
    ) if results else "No context."
    user_prompt = f"{system_prompt}\n\nContext:\n{context_block}\n\nAnalytics Question:\n{req.query}\n\nProvide structured insights."

    model = genai.GenerativeModel("gemini-2.5-flash")
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

    analysis = ""
    if resp and getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        if cand.finish_reason == "SAFETY":
            analysis = "Response blocked by Gemini safety filters."
        elif hasattr(cand, "content") and cand.content:
            if hasattr(cand.content, "parts"):
                texts = [getattr(p, "text", "") for p in cand.content.parts]
                analysis = "".join([t for t in texts if t])
            elif isinstance(cand.content, str):
                analysis = cand.content

    if not analysis:
        analysis = "No analytics could be generated from the current context."

    return AnalyticsResponse(analysis=analysis, sources=sources)

# ----------------------------
# NEW: NLP Rule Create Endpoint
# ----------------------------

@app.post("/nlp_rule_create", response_model=List[RuleModel])
def nlp_rule_create(req: NlpRuleCreateRequest, _: None = Depends(check_auth)):
    # 1) Parse rules from NL using Gemini
    rules = parse_rules_with_gemini(req)

    # 2) Validate against schema columns and attach warnings already handled in parse
    # 3) Save each rule under <PROJ_ROOT>/dw/rule/*.json
    for i, rule in enumerate(rules):
        fpath = save_rule_to_disk(rule)
        rules[i].source_path = fpath

        # 4) Index in Qdrant with metadata.source_type="dw_rules"
        try:
            index_rule_in_qdrant(rule)
        except Exception as e:
            # Keep rule but add warning
            w = rules[i].warnings or []
            w.append(f"Indexing failed: {e}")
            rules[i].warnings = w

    # 5) Optionally auto-commit (already persisted to disk + indexed)
    # Nothing else needed server-side; client can toggle active via /upsert_batch if desired.
    return rules
