
# DQ Relay (FastAPI) � unified on google-genai
import os
import time
import hashlib
import uuid
import json
import re
from functools import lru_cache

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import re
from dateutil.parser import parse as dt_parse  # add python-dateutil to requirements if not present
from typing import List, Optional, Dict, Any
from enum import Enum


# NEW SDK (one SDK for gen content + structured outputs + embeddings)
from google import genai
from google.genai import types

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

# -----------------------------------------------------------------------------
# Env & configuration
# -----------------------------------------------------------------------------
REQUIRED_ENV = [
    "GEMINI_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
]
for var in REQUIRED_ENV:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # or GOOGLE_API_KEY
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dq_docs")

# Default embedding model: 768 dims � matches Qdrant dim below
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# Default gen model for text answers
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")  # optional bearer token
PROJ_ROOT = os.getenv("PROJ_ROOT", ".")  # root for saving dw/rule/*.json

# Ensure dw/rule directory exists
DW_RULE_DIR = os.path.join(PROJ_ROOT, "dq", "rule")
os.makedirs(DW_RULE_DIR, exist_ok=True)

# One client for all operations
genai_client = genai.Client()  # auto picks GEMINI_API_KEY / GOOGLE_API_KEY

# -----------------------------------------------------------------------------
# Qdrant client
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection(client: QdrantClient, dim: int = 768):
    # If you move to gemini-embedding-001 (up to 3072 dims), change this
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
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
    filters: Optional[Dict[str, Any]] = None
    max_output_tokens: int = 1024
    temperature: float = 0.2

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class RecommendationRequest(BaseModel):
    profile_summary: str = Field(..., description="Data profile summary text from Streamlit app")
    max_output_tokens: int = 1024
    temperature: float = 0.3

class RecommendationResponse(BaseModel):
    recommendations: str

class AnalyticsRequest(BaseModel):
    query: str = Field(..., description="Analytics question, e.g. 'Top 10 failed dq rules for last 3 months'")
    top_k: int = 20
    filters: Optional[Dict[str, Any]] = None
    max_output_tokens: int = 1024
    temperature: float = 0.2

class AnalyticsResponse(BaseModel):
    analysis: str
    sources: List[Dict[str, Any]]

# NLP Rule Creation (same shape)
ALLOWED_TYPES = ['not_null', 'regex', 'domain', 'range', 'unique', 'cross_field', 'freshness', 'referential']

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

# Smart routing
class Intent(str, Enum):
    chat = "chat"
    analytics = "analytics"
    rule_create = "rule_create"

class SmartRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    schema: Optional["SchemaInput"] = None
    top_k: int = 12
    temperature: float = 0.2
    max_output_tokens: int = 1024

class SmartResponse(BaseModel):
    intent: Intent
    answer: Optional[str] = None
    analysis: Optional[str] = None
    # rules omitted for now
    sources: List[Dict[str, Any]] = []

class IntentChoice(BaseModel):
    intent: Intent

# ----------------------------
# Models to match your UI
# ----------------------------
class Operator(str, Enum):
    contains = "contains"
    is_ = "is"
    is_not = "is not"
    is_within = "is within"
    is_not_within = "is not within"
    is_less_than = "is less than"
    is_less_equal = "is less than or equal to"
    is_greater_than = "is greater than"
    is_greater_equal = "is greater than or equal to"

class ConditionType(str, Enum):
    null_value = "null value"
    string_value = "string value"
    integer_value = "integer value"
    float_value = "float value"
    current_timestamp = "current timestamp"
    expression = "expression"
    function = "function"  # for is within / is not within

class Statement(BaseModel):
    Column: str
    Operator: Operator
    Condition_Type: ConditionType
    Condition_Value: Optional[str] = None
    # Optional hint to drive your popover type ("String" | "Float/Integer" | "Date/Time")
    DType: Optional[str] = Field(default=None)

class InputColumn(BaseModel):
    name: str
    description: Optional[str] = None
    data_type: str = "String"        # "String" | "Integer" | "Float" | "Date/Time"
    max_length: Optional[str] = ""

class RuleMapResponse(BaseModel):
    rule_name: Optional[str] = None
    rule_details: Optional[str] = None
    inputs: List[InputColumn]
    groups: List[List[Statement]]     # same shape as Streamlit Rule Builder

class NlpRuleMapRequest(BaseModel):
    text: str
    schema: Dict[str, Any]            # Expected shape of your SchemaInput


# -----------------------------------------------------------------------------
# Auth dependency
# -----------------------------------------------------------------------------
def check_auth(authorization: Optional[str] = Header(None)):
    if AUTH_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.split(" ", 1)[1]
        if token != AUTH_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden")

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def stable_id(text: str, metadata: Optional[Metadata]) -> str:
    base = (text or "") + "\n" + (metadata.source_type if metadata and metadata.source_type else "")
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
    """Batch embeddings via google-genai."""
    if not texts:
        return []
    result = genai_client.models.embed_content(model=EMBED_MODEL, contents=texts)
    # result.embeddings is a list; each item has .values
    return [emb.values for emb in result.embeddings]

def embed_query(text: str) -> List[float]:
    """Single query embedding via google-genai."""
    result = genai_client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return result.embeddings[0].values

def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not filters:
        return None
    conditions = []
    for k, v in filters.items():
        conditions.append(FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v)))
    return Filter(should=conditions) if conditions else None

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def rule_to_text(rule: RuleModel) -> str:
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
    return "".join([c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in (name or "")])[:80]

def _strip_code_fences(text: str) -> str:
    """Safely remove ```json ... ``` fences."""
    return re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', (text or "").strip(), flags=re.IGNORECASE)

# ----------------------------
# Helpers (synonyms & dtype inference)
# ----------------------------
_OP_SYNONYMS = {
    "equals": "is",
    "==": "is",
    "!=": "is not",
    "not equals": "is not",
    "in": "is within",
    "not in": "is not within",
    ">": "is greater than",
    ">=": "is greater than or equal to",
    "<": "is less than",
    "<=": "is less than or equal to",
}

def _norm_op(txt: str) -> str:
    t = (txt or "").strip().lower()
    return _OP_SYNONYMS.get(t, t)

def _is_iso_date(s: str) -> bool:
    # Lightweight date heuristic (avoid extra dependencies)
    s = (s or "").strip()
    # yyyy-mm-dd or yyyy/mm/dd; extend as needed
    return bool(re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", s))

def _infer_numeric_type(s: str) -> Optional[str]:
    val = (s or "").strip()
    # strictly integer?
    if re.fullmatch(r"[+-]?\d+", val):
        return "integer value"
    # float?
    if re.fullmatch(r"[+-]?\d+\.\d+", val):
        return "float value"
    return None

def _infer_scalar_dtype_hint(s: str) -> str:
    """Return UI popover dtype hint: 'Float/Integer' | 'Date/Time' | 'String'."""
    if _infer_numeric_type(s):
        return "Float/Integer"
    if _is_iso_date(s):
        return "Date/Time"
    return "String"

def _infer_list_dtype_hint(values: List[str]) -> str:
    votes = {"String": 0, "Float/Integer": 0, "Date/Time": 0}
    for v in values or []:
        votes[_infer_scalar_dtype_hint(v)] += 1
    return max(votes, key=votes.get) if votes else "String"

def _split_list(raw: str) -> List[str]:
    s = (raw or "").strip().strip("{}")
    items = [i.strip() for i in re.split(r"[,\|]", s) if i.strip()]
    return items

def _to_ui_dtype(dtype_hint: str) -> str:
    """Map our hint to Input Columns choices: String | Integer | Float | Date/Time."""
    if dtype_hint == "Float/Integer":
        # default to Integer; user can adjust to Float
        return "Integer"
    if dtype_hint == "Date/Time":
        return "Date/Time"
    return "String"

# -----------------------------------------------------------------------------
# NLP parsing � updated to new SDK (still plain JSON parsing)
# -----------------------------------------------------------------------------
def parse_rules_with_gemini(req: NlpRuleCreateRequest) -> List[RuleModel]:
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
    resp = genai_client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=1024, temperature=0.1),
    )
    text = getattr(resp, "text", "") or ""
    if not text and getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
            text = "".join(getattr(p, "text", "") for p in cand.content.parts if hasattr(p, "text"))
    if not text:
        raise HTTPException(status_code=500, detail="Model returned no rules")
    try:
        cleaned = _strip_code_fences(text)
        raw_rules = json.loads(cleaned)
        if not isinstance(raw_rules, list):
            raise ValueError("Expected a JSON array of rules")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse rules JSON: {e}")

    rules: List[RuleModel] = []
    for r in raw_rules:
        try:
            target = r.get("target", {}) or {}
            target.setdefault("dataset_alias", req.schema.dataset_alias)
            target.setdefault("path_or_table", req.schema.path_or_table)

            rid = gen_rule_id()
            created_at = now_iso()
            warnings = r.get("warnings", []) or []

            col = target.get("column")
            if col and col not in req.schema.column_names:
                warnings.append(f"Column '{col}' not found in schema; please fix.")

            pred = r.get("predicate", {}) or {}
            ptype = pred.get("type")
            if ptype not in ALLOWED_TYPES:
                warnings.append(f"Unsupported type '{ptype}'. Allowed: {ALLOWED_TYPES}")

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
            raise HTTPException(status_code=400, detail=f"Rule validation error: {e}")
    return rules

def save_rule_to_disk(rule: RuleModel) -> str:
    fname = sanitize_filename(rule.name or rule.id) + ".json"
    fpath = os.path.join(DW_RULE_DIR, fname)
    payload = rule.dict()
    payload["metadata"] = {
        "source_type": "dw_rules",
        "path_or_table": rule.target.path_or_table,
        "source_name": rule.name
    }
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

# -----------------------------------------------------------------------------
# FastAPI app & endpoints
# -----------------------------------------------------------------------------
app = FastAPI(title="DQ Relay", version="1.2.0")

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
    # text-embedding-004 => 768 dims; if you use gemini-embedding-001, raise dim
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
        for chunk, vec in zip(chunks, vectors):
            pid = item.id or str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=pid,
                    vector={"default": vec},
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
        f"[{r.payload.get('metadata', {}).get('source_type','doc')}] "
        f"{r.payload.get('metadata', {}).get('source_name','')} "
        f"{r.payload.get('metadata', {}).get('path_or_table','')}\n"
        f"{r.payload.get('text','')}"
        for r in results
    ) if results else "No context."

    user_prompt = (
        f"{system_prompt}\n\nContext:\n{context_block}\n\nUser question:\n{req.query}\n\n"
        "When you cite or refer, mention the source_type or rule names if present."
    )

    resp = genai_client.models.generate_content(
        model=GEN_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=req.max_output_tokens,
            temperature=req.temperature
        )
    )
    answer = getattr(resp, "text", "") or ""
    if not answer and getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
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
    resp = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=req.max_output_tokens,
            temperature=req.temperature
        )
    )
    answer = getattr(resp, "text", "") or ""
    if not answer and getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        if getattr(cand, "content", None):
            if getattr(cand.content, "parts", None):
                texts = [getattr(p, "text", "") for p in cand.content.parts]
                answer = "".join([t for t in texts if t])
            elif isinstance(cand.content, str):
                answer = cand.content
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
        f"[{r.payload.get('metadata', {}).get('source_type','doc')}] "
        f"{r.payload.get('metadata', {}).get('source_name','')} "
        f"{r.payload.get('metadata', {}).get('path_or_table','')}\n"
        f"{r.payload.get('text','')}"
        for r in results
    ) if results else "No context."
    user_prompt = (
        f"{system_prompt}\n\nContext:\n{context_block}\n\nAnalytics Question:\n{req.query}\n\nProvide structured insights."
    )

    resp = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=req.max_output_tokens,
            temperature=req.temperature
        )
    )
    analysis = getattr(resp, "text", "") or ""
    if not analysis and getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
            texts = [getattr(p, "text", "") for p in cand.content.parts]
            analysis = "".join([t for t in texts if t])
        elif isinstance(cand.content, str):
            analysis = cand.content
    if not analysis:
        analysis = "No analytics could be generated from the current context."
    return AnalyticsResponse(analysis=analysis, sources=sources)

@app.post("/nlp_rule_create", response_model=List[RuleModel])
def nlp_rule_create(req: NlpRuleCreateRequest, _: None = Depends(check_auth)):
    rules = parse_rules_with_gemini(req)
    for i, rule in enumerate(rules):
        fpath = save_rule_to_disk(rule)
        rules[i].source_path = fpath
        try:
            index_rule_in_qdrant(rule)
        except Exception as e:
            w = rules[i].warnings or []
            w.append(f"Indexing failed: {e}")
            rules[i].warnings = w
    return rules

# -----------------------------------------------------------------------------
# Smart endpoint: Structured Outputs intent + delegation
# -----------------------------------------------------------------------------
def _classify_intent_structured(query: str) -> Intent:
    try:
        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Classify into intent (chat|analytics|rule_create): {query}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IntentChoice,
                temperature=0.0
            )
        )
        return resp.parsed.intent
    except Exception:
        # fallback heuristics
        q = (query or "").lower()
        if ("create rule" in q) or ("define rule" in q):
            return Intent.rule_create
        rule_words = ["not null","regex","matches","unique","referential","exists in",
                      "freshness","within","between","domain","range",">=","<=",
                      ">", "<", "=", "!=", "satisfies"]
        if any(w in q for w in rule_words):
            return Intent.rule_create
        analytics_words = ["top ","trend","last week","last month","last 3 months",
                           "distribution","aggregate","summary","count of failures",
                           "most missing","percent null","grouped by","failed rules"]
        if any(w in q for w in analytics_words) or q.startswith("top "):
            return Intent.analytics
        return Intent.chat

@app.post("/smart", response_model=SmartResponse)
def smart(req: SmartRequest, _: None = Depends(check_auth)):
    intent = _classify_intent_structured(req.query)

    if intent == Intent.chat:
        resp = chat(ChatRequest(
            query=req.query,
            top_k=max(1, req.top_k),
            filters=req.filters,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens
        ))
        return SmartResponse(intent=intent, answer=resp.answer, sources=resp.sources)

    if intent == Intent.analytics:
        resp = analytics(AnalyticsRequest(
            query=req.query,
            top_k=max(20, req.top_k),
            filters=req.filters,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens
        ))
        return SmartResponse(intent=intent, analysis=resp.analysis, sources=resp.sources)

    # For now, we return guidance for rule_create (UI path disabled)
    return SmartResponse(
        intent=intent,
        answer="Intent classified as 'rule_create'. Client currently has rule creation disabled.",
        sources=[]
    )

# ----------------------------
# The endpoint
# ----------------------------
@app.post("/nlp_rule_map", response_model=RuleMapResponse)
def nlp_rule_map(req: NlpRuleMapRequest, _: None = Depends(check_auth)):
    try:
        text = (req.text or "").strip()
        schema_cols = req.schema.get("columns", [])
        schema_cols_str = ", ".join([f"{c.get('name')}:{c.get('dtype','')}" for c in schema_cols if c.get("name")])

        operator_list = [
            "contains", "is", "is not", "is within", "is not within",
            "is less than", "is less than or equal to",
            "is greater than", "is greater than or equal to",
        ]
        prompt = (
            "Convert natural language DQ requirements into Rule Builder statements.\n"
            f"Valid operators: {operator_list}\n"
            "Condition_Type must be allowed for the chosen operator.\n"
            "ANDs go into a single group; new group == OR.\n"
            "‘between A and B’ MUST become = A and = B on the same column.\n"
            "‘in {a,b,c}’, ‘one of’, ‘is within’ => operator 'is within' and Condition_Type 'function'; "
            "‘not in’ => 'is not within'.\n"
            "Use EXACT schema column names where applicable (do not invent columns).\n"
            "Include optional rule_name and rule_details if obvious; otherwise leave empty.\n\n"
            f"Schema columns: [{schema_cols_str}]\n\n"
            f"User input:\n{text}\n\n"
            "Return JSON matching: { rule_name?, rule_details?, inputs?, groups? }."
        )

        # 1) Try Structured Outputs
        try:
            so = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RuleMapResponse,
                    temperature=0.0, max_output_tokens=1024
                )
            )
            parsed: RuleMapResponse = so.parsed
            raw_groups = [ [ s.model_dump() for s in group ] for group in parsed.groups ]
            groups = _normalize_groups(raw_groups)
            # build inputs from groups (prefer stronger types if repeated)
            seen: Dict[str, InputColumn] = {}
            rank = {"Date/Time":3, "Float":2, "Integer":2, "String":1}
            for g in groups:
                for s in g:
                    col = (s.Column or "").strip()
                    if not col:
                        continue
                    hint = s.DType or "String"
                    dtype = _to_ui_dtype(hint)
                    cur = seen.get(col)
                    if not cur:
                        seen[col] = InputColumn(name=col, data_type=dtype, description="", max_length="")
                    else:
                        if rank.get(dtype,1) > rank.get(cur.data_type,1):
                            cur.data_type = dtype
            inputs = list(seen.values())
            return RuleMapResponse(
                rule_name=parsed.rule_name,
                rule_details=parsed.rule_details,
                inputs=inputs,
                groups=groups
            )
        except Exception as so_err:
            # 2) Fallback: no-schema generation, then coerce
            ns = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0, max_output_tokens=1024
                )
            )
            raw_text = getattr(ns, "text", "") or ""
            if not raw_text and getattr(ns, "candidates", None):
                cand = ns.candidates[0]
                if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                    raw_text = "".join(getattr(p, "text", "") for p in cand.content.parts if hasattr(p, "text"))
            if not raw_text:
                raise HTTPException(status_code=400, detail="Model returned no JSON.")
            # strip code fences if any
            cleaned = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw_text.strip(), flags=re.IGNORECASE)
            try:
                data = json.loads(cleaned)
            except Exception as je:
                raise HTTPException(status_code=400, detail=f"Invalid JSON from model: {je}")

            # data can be missing fields; make them optional
            rule_name    = (data.get("rule_name") or data.get("name") or "")
            rule_details = (data.get("rule_details") or data.get("details") or "")
            raw_groups   = data.get("groups") or data.get("statements") or []
            groups = _normalize_groups(raw_groups)

            # build inputs
            seen: Dict[str, InputColumn] = {}
            rank = {"Date/Time":3, "Float":2, "Integer":2, "String":1}
            for g in groups:
                for s in g:
                    col = (s.Column or "").strip()
                    if not col:
                        continue
                    hint = s.DType or "String"
                    dtype = _to_ui_dtype(hint)
                    cur = seen.get(col)
                    if not cur:
                        seen[col] = InputColumn(name=col, data_type=dtype, description="", max_length="")
                    else:
                        if rank.get(dtype,1) > rank.get(cur.data_type,1):
                            cur.data_type = dtype
            inputs = list(seen.values())
            return RuleMapResponse(
                rule_name=rule_name or None,
                rule_details=rule_details or None,
                inputs=inputs,
                groups=groups
            )

    except HTTPException:
        raise
    except ValidationError as ve:
        # Pydantic typed error
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"NLP mapping failed: {e}")
