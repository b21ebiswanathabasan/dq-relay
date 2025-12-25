
# DQ Relay (FastAPI) unified on google-genai — UPDATED (11-Nov-2025)

import os
import time
import hashlib
import uuid
import json
import re
from functools import lru_cache
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel, Field, validator

from dateutil.parser import parse as dt_parse  # ensure python-dateutil in requirements

# NEW SDK (one SDK for gen content + structured outputs + embeddings)
from google import genai
from google.genai import types

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

# ------------------------------------------------------------------------------------
# Env & configuration
# ------------------------------------------------------------------------------------
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

# Default embedding model: 768 dims — matches Qdrant dim below
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


# ------------------------------------------------------------------------------------
# Qdrant client
# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------------
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

# Models to match your UI
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
    data_type: str = "String"  # "String" | "Integer" | "Float" | "Date/Time"
    max_length: Optional[str] = ""

class RuleMapResponse(BaseModel):
    rule_name: Optional[str] = None
    rule_details: Optional[str] = None
    inputs: List[InputColumn]
    groups: List[List[Statement]]  # same shape as Streamlit Rule Builder
    dimension: Optional[str] = None
    dimension: Optional[str] = None

class NlpRuleMapRequest(BaseModel):
    text: str
    schema: Dict[str, Any]  # Expected shape of your SchemaInput

# ------------------------------------------------------------------------------------
# Auth dependency
# ------------------------------------------------------------------------------------
def check_auth(authorization: Optional[str] = Header(None)):
    if AUTH_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.split(" ", 1)[1]
        if token != AUTH_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden")

# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------
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

# --- UPDATED: safer fence stripper ---
def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()

# Operator synonyms
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
    # Lightweight date heuristic (yyyy-mm-dd or yyyy/mm/dd)
    s = (s or "").strip()
    return bool(re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", s))

def _infer_numeric_type(s: str) -> Optional[str]:
    val = (s or "").strip()
    # strictly integer?
    if re.fullmatch(r"[+\-]?\d+", val):
        return "integer value"
    # float?
    if re.fullmatch(r"[+\-]?\d+\.\d+", val):
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

# --- UPDATED: simpler comma splitter ---
def _split_list(raw: str) -> List[str]:
    s = (raw or "").strip().strip("{}")
    return [i.strip() for i in s.split(",") if i.strip()]

# --- NEW: Condition type normalizer ---
def _norm_condition_type(s: str) -> str:
    t = (s or "").strip().lower()
    if t in {"string", "text"}: t = "string value"
    if t in {"integer", "int"}: t = "integer value"
    if t in {"float", "double", "decimal"}: t = "float value"
    if t in {"timestamp", "now", "today", "current time"}: t = "current timestamp"
    if t in {"expr", "expression"}: t = "expression"
    if t in {"null", "none"}: t = "null value"
    return t

def _to_ui_dtype(dtype_hint: str) -> str:
    """Map our hint to Input Columns choices: String | Integer | Float | Date/Time."""
    if dtype_hint == "Float/Integer":
        # default to Integer; user can adjust to Float
        return "Integer"
    if dtype_hint == "Date/Time":
        return "Date/Time"
    return "String"

# ------------------------------------------------------------------------------------
# NLP parsing — updated to new SDK (still plain JSON parsing for /nlp_rule_create)
# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
# FastAPI app & endpoints
# ------------------------------------------------------------------------------------
app = FastAPI(title="DQ Relay", version="1.3.0")

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
    # Detect mode tag in the incoming summary
    mode = "SUMMARY"
    m = re.search(r"\[MODE:\s*([A-Z_]+)\s*\]", req.profile_summary or "", flags=re.IGNORECASE)
    if m:
        mode = m.group(1).upper()

    if mode == "DQ_RULES":
        prompt = _strong_json_rules_prompt(req.profile_summary)
        json_text = _gen_strict_json(
            genai_client, model=os.getenv("GEN_MODEL", "gemini-2.5-flash"),
            prompt=prompt, max_tokens=req.max_output_tokens, temperature=0.2
        )
        if not json_text or not json_text.strip().startswith("{"):
            # Make it unambiguous for the client to fall back if needed
            json_text = "{}"
        return RecommendationResponse(recommendations=json_text)

    if mode == "CLEANSING":
        prompt = _strong_json_cleanse_prompt(req.profile_summary)
        json_text = _gen_strict_json(
            genai_client, model=os.getenv("GEN_MODEL", "gemini-2.5-flash"),
            prompt=prompt, max_tokens=req.max_output_tokens, temperature=0.2
        )
        if not json_text or not json_text.strip().startswith("{"):
            json_text = "{}"
        return RecommendationResponse(recommendations=json_text)

    # Default path: narrative markdown summary (unchanged)
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


# --- Add these helpers near /recommend ---

def _strong_json_rules_prompt(profile_summary: str) -> str:
    example = (
        "{\n"
        '  "colA": ["NOT_NULL","UNIQUE", {"values":["A","B"]}, "DOMAIN_SET"],\n'
        '  "colB": ["NOT_NULL","REGEX_MATCH:^\\d{4}-\\d{2}-\\d{2}$","DATE_PARSABLE"]\n'
        "}"
    )
    return (
        "You are a Data Quality rule designer. Return ONLY STRICT JSON—no prose, no markdown, no comments.\n"
        "JSON must be a single object mapping column-name → array of items:\n"
        "  • Simple constraints as strings: NOT_NULL, UNIQUE, DOMAIN_SET, DATE_PARSABLE, REGEX_MATCH:<pattern>\n"
        "  • For domain sets, include an object {\"values\": [\"v1\",\"v2\",...]} alongside \"DOMAIN_SET\".\n"
        f"Example:\n{example}\n\n"
        "Constraints MUST be inferred only from the provided digest/columns.\n"
        "Dataset Profile Summary:\n"
        f"{profile_summary}\n"
    )

def _strong_json_cleanse_prompt(profile_summary: str) -> str:
    example = (
        "{\n"
        '  "name": ["Trim leading/trailing spaces", "Uppercase canonical form"],\n'
        '  "age": ["Impute missing with median", "Cap outliers at P99"],\n'
        '  "email": ["Regex validate", "Lowercase before compare"]\n'
        "}"
    )
    return (
        "You are a Data Cleansing planner. Return ONLY STRICT JSON—no prose, no markdown, no comments.\n"
        "JSON must be a single object mapping column-name → array of short actionable steps.\n"
        f"Example:\n{example}\n\n"
        "Base your plan strictly on the provided digest/columns.\n"
        "Dataset Profile Summary:\n"
        f"{profile_summary}\n"
    )

def _gen_strict_json(client, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    # Prefer structured JSON text; keep SDK simple & robust
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
    )
    text = getattr(resp, "text", "") or ""
    if not text and getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
            text = "".join(getattr(p, "text", "") for p in cand.content.parts if hasattr(p, "text"))
    return _strip_code_fences(text).strip()

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

# ------------------------------------------------------------------------------------
# Smart endpoint: Structured Outputs intent + delegation
# ------------------------------------------------------------------------------------
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
        analytics_words = ["top ", "trend", "last week", "last month", "last 3 months",
                           "distribution", "aggregate", "summary", "count of failures",
                           "most missing", "percent null", "grouped by", "failed rules"]
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


# --- Helpers for robust fallback -------------------------------------------------

def _split_items_natural(raw: str) -> List[str]:
    """
    Split a natural-language list like:
      "Active, Live", "Active and Live", "Active & Live", "{Active, Live}"
    into ["Active", "Live"].
    """
    s = (raw or "").strip()
    s = s.strip("{}")
    # Normalize common conjunctions to comma
    s = re.sub(r"\s+(?:and|AND)\s+", ",", s)
    s = s.replace("&", ",")
    # Final split on comma
    items = [i.strip() for i in s.split(",") if i.strip()]
    return items

def _heuristic_rulemap_from_text(text: str, schema_cols: List[Dict[str, Any]]) -> Optional[RuleMapResponse]:
    """
    Very small heuristic to cover common NL patterns when the model returns no JSON:
    - "<col> between 30 & 60"
    - "<col> in Active, Live" / "<col> in {Active, Live}" / "<col> in Active and Live"
    - "Ensure '<col>' is unique / no duplicates"
    Produces one group with AND across all detected statements.
    """
    cols_set = {c.get("name") for c in (schema_cols or []) if c.get("name")}
    t = text or ""
    lower = t.lower()

    groups: List[List[Statement]] = []
    statements: List[Statement] = []
    inputs_map: Dict[str, InputColumn] = {}

    # Try to spot columns explicitly quoted in NL (e.g., 'customer_id')
    quoted_cols = set(re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']", t))

    # (1) between: "<col> between A & B"  -- allow numbers (int/float) or dates in future
    m_between = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\b[^.\n]*\bbetween\b\s*([+-]?\d+(?:\.\d+)?)\s*&\s*([+-]?\d+(?:\.\d+)?)",
        t, flags=re.IGNORECASE
    )
    if m_between:
        col = m_between.group(1)
        a = m_between.group(2)
        b = m_between.group(3)
        if (not cols_set) or (col in cols_set) or (col in quoted_cols):
            statements.append(Statement(
                Column=col,
                Operator=Operator.is_greater_equal,
                Condition_Type=ConditionType.integer_value if re.fullmatch(r"[+-]?\d+", a) else ConditionType.float_value,
                Condition_Value=a,
                DType="Float/Integer"
            ))
            statements.append(Statement(
                Column=col,
                Operator=Operator.is_less_equal,
                Condition_Type=ConditionType.integer_value if re.fullmatch(r"[+-]?\d+", b) else ConditionType.float_value,
                Condition_Value=b,
                DType="Float/Integer"
            ))
            inputs_map[col] = InputColumn(name=col, data_type="Integer", description="", max_length="")

    # (2) within: "<col> in Active, Live" or "{Active, Live}" or "Active and Live"
    m_within = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\b[^.\n]*\bin\b\s*\{?([A-Za-z0-9_ ,&]+)\}?",
        t, flags=re.IGNORECASE
    )
    if m_within:
        col = m_within.group(1)
        raw = m_within.group(2)
        items = _split_items_natural(raw)
        if items and ((not cols_set) or (col in cols_set) or (col in quoted_cols)):
            statements.append(Statement(
                Column=col,
                Operator=Operator.is_within,
                Condition_Type=ConditionType.function,
                Condition_Value=", ".join(items),
                DType="String"
            ))
            inputs_map[col] = InputColumn(name=col, data_type="String", description="", max_length="")

    # (3) uniqueness: "<col> must be unique" or "no duplicates"
    # We scan for either a quoted col or any schema column in the text near uniqueness keywords.
    uniq_hits = set()
    uniq_keywords = r"(unique|no\s+duplicates|distinct)"
    # Quoted column occurrence
    for qc in quoted_cols:
        if re.search(rf"\b{re.escape(qc)}\b[^.\n]*\b{uniq_keywords}\b", lower):
            uniq_hits.add(qc)
    # Any schema column occurrence
    for sc in (cols_set or []):
        if re.search(rf"\b{re.escape(sc)}\b[^.\n]*\b{uniq_keywords}\b", lower):
            uniq_hits.add(sc)
    # If no schema was provided, try to guess a token before 'column'
    if not uniq_hits and not cols_set:
        m_guess = re.search(r"['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s+column\b[^.\n]*\b(unique|no\s+duplicates|distinct)\b", lower)
        if m_guess:
            uniq_hits.add(m_guess.group(1))

    for col in uniq_hits:
        statements.append(Statement(
            Column=col,
            Operator=Operator.is_,  # drive expression under 'is'
            Condition_Type=ConditionType.expression,
            Condition_Value=f"CountDistinct({col}) = Count({col})",
            DType="String"
        ))
        inputs_map[col] = InputColumn(name=col, data_type="String", description="", max_length="")

    if statements:
        groups = [statements]  # single AND-group
        return RuleMapResponse(
            rule_name="",
            rule_details="",
            inputs=list(inputs_map.values()),
            groups=groups
        )
    return None


# --- Endpoint: /nlp_rule_map ----------------------------------------------------

@app.post("/nlp_rule_map", response_model=RuleMapResponse)
def nlp_rule_map(req: NlpRuleMapRequest, _: None = Depends(check_auth)):
    """
    Convert NL into UI-ready Rule Builder groups & inferred Input Columns.
    - Structured Outputs first (Pydantic schema).
    - Robust fallback (no 400) + heuristics for common NL (between, lists, uniqueness).
    - Post-normalization: 'between' -> >= and <=, 'is within' -> function list, etc.
    - Trim/strip hardener upgrades string comparisons to expression with Trim+Upper.
    - Predicts 'dimension' via LLM; if missing, uses heuristics; always returns 'dimension'.
    """
    try:
        text: str = (req.text or "").strip()
        schema_dict: Dict[str, Any] = req.schema or {}
        columns_in_schema: List[Dict[str, Any]] = schema_dict.get("columns", []) or []
        schema_cols_str = ", ".join(
            [f"{c.get('name')}: {c.get('dtype','')}" for c in columns_in_schema if c.get("name")]
        )

        # --- Tiny helpers local to this function --------------------------------
        def _split_items_natural(raw: str) -> List[str]:
            s = (raw or "").strip().strip("{}")
            s = re.sub(r"\s+(?:and|AND)\s+", ",", s)
            s = s.replace("&", ",")
            return [i.strip() for i in s.split(",") if i.strip()]

        def _heuristic_rulemap_from_text(_t: str, _schema_cols: List[Dict[str, Any]]) -> Optional[RuleMapResponse]:
            cols_set = {c.get("name") for c in (_schema_cols or []) if c.get("name")}
            lower = (_t or "").lower()
            statements: List[Statement] = []
            inputs_map: Dict[str, InputColumn] = {}

            # between: "<col> between 30 & 60"
            m_between = re.search(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\b[^.\n]*\bbetween\b\s*([+-]?\d+(?:\.\d+)?)\s*&\s*([+-]?\d+(?:\.\d+)?)",
                _t, flags=re.IGNORECASE
            )
            if m_between:
                col, a, b = m_between.group(1), m_between.group(2), m_between.group(3)
                if (not cols_set) or (col in cols_set):
                    statements += [
                        Statement(Column=col, Operator=Operator.is_greater_equal,
                                  Condition_Type=ConditionType.integer_value if re.fullmatch(r"[+-]?\d+", a) else ConditionType.float_value,
                                  Condition_Value=a, DType="Float/Integer"),
                        Statement(Column=col, Operator=Operator.is_less_equal,
                                  Condition_Type=ConditionType.integer_value if re.fullmatch(r"[+-]?\d+", b) else ConditionType.float_value,
                                  Condition_Value=b, DType="Float/Integer")
                    ]
                    inputs_map[col] = InputColumn(name=col, data_type="Integer", description="", max_length="")

            # within: "<col> in Active, Live"
            m_within = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b[^.\n]*\bin\b\s*\{?([A-Za-z0-9_ ,&]+)\}?", _t, flags=re.IGNORECASE)
            if m_within:
                col, raw = m_within.group(1), m_within.group(2)
                items = _split_items_natural(raw)
                if items and ((not cols_set) or (col in cols_set)):
                    statements.append(Statement(Column=col, Operator=Operator.is_within,
                                               Condition_Type=ConditionType.function,
                                               Condition_Value=", ".join(items), DType="String"))
                    inputs_map[col] = InputColumn(name=col, data_type="String", description="", max_length="")

            # uniqueness: "<col> must be unique" or "no duplicates"
            uniq_hits = set()
            for sc in (cols_set or []):
                if re.search(rf"\b{re.escape(sc)}\b[^.\n]*\b(unique|no\s+duplicates|distinct)\b", lower):
                    uniq_hits.add(sc)
            for col in uniq_hits:
                statements.append(Statement(
                    Column=col, Operator=Operator.is_, Condition_Type=ConditionType.expression,
                    Condition_Value=f"IsUnique({col})", DType="String"   # <-- Template uses IsUnique()
                ))
                inputs_map[col] = InputColumn(name=col, data_type="String", description="", max_length="")

            if statements:
                return RuleMapResponse(rule_name="", rule_details="", inputs=list(inputs_map.values()), groups=[statements], dimension=None)
            return None
        # ------------------------------------------------------------------------

        operator_list = [
            "contains", "is", "is not", "is within", "is not within",
            "is less than", "is less than or equal to", "is greater than", "is greater than or equal to",
        ]

        prompt = (
            "You convert natural language DQ requirements into Rule Builder statements.\n"
            f"Valid operators: {operator_list}\n"
            "Condition_Type must be one of the allowed types for the chosen operator.\n"
            "Rules:\n"
            " - Use AND inside a group; start a new group for OR.\n"
            " - 'between A and B' MUST become two statements on the same Column:\n"
            "   one with 'is greater than or equal to' A, and one with 'is less than or equal to' B.\n"
            " - 'in {a,b,c}', 'one of', 'is within' map to operator 'is within' and Condition_Type 'function'.\n"
            " - 'not in' -> 'is not within'.\n"
            " - 'equals' -> 'is'; 'not equals' -> 'is not'; symbols (> ,>= ,< ,<=) map to the corresponding operators.\n"
            " - Use the EXACT column names given in schema—do not invent columns.\n"
            " - If the text mentions trimming/stripping spaces, emit an expression that applies Trim/LTrim/RTrim BEFORE comparison.\n"
            " - For list values, keep them as a comma-separated string in Condition_Value (e.g., \"A, B, C\").\n"
            " - Include rule_name and rule_details if implied by the text; else leave them empty.\n"
            " - Return ONLY JSON, no markdown, no comments, no trailing commas; keys/strings must use double-quotes.\n\n"
            f"Schema columns: [{schema_cols_str}]\n\n"
            f"User input:\n{text}\n\n"
            "Produce JSON matching RuleMapResponse."
        )

        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RuleMapResponse,  # Structured Outputs
                temperature=0.0,
                max_output_tokens=1024
            )
        )

        # --- Fallback (no 400) ---
        if getattr(resp, "parsed", None) is None:
            raw_text = getattr(resp, "text", "") or ""
            if (not raw_text.strip()) and getattr(resp, "candidates", None):
                try:
                    cand = resp.candidates[0]
                    parts = getattr(getattr(cand, "content", None), "parts", None)
                    if parts:
                        raw_text = "".join(
                            getattr(p, "text", "") for p in parts
                            if hasattr(p, "text") and isinstance(getattr(p, "text"), str)
                        )
                except Exception:
                    pass

            if not raw_text.strip():
                resp2 = genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=1024, temperature=0.0)
                )
                raw_text = getattr(resp2, "text", "") or ""
                if (not raw_text.strip()) and getattr(resp2, "candidates", None):
                    try:
                        cand = resp2.candidates[0]
                        parts = getattr(getattr(cand, "content", None), "parts", None)
                        if parts:
                            raw_text = "".join(
                                getattr(p, "text", "") for p in parts
                                if hasattr(p, "text") and isinstance(getattr(p, "text"), str)
                            )
                    except Exception:
                        pass

            txt = _strip_code_fences(raw_text).strip()

            if not txt:
                heur = _heuristic_rulemap_from_text(text, columns_in_schema)
                parsed = heur if heur else RuleMapResponse(
                    rule_name="", rule_details="", inputs=[],
                    groups=[[Statement(Column="", Operator=Operator.is_,
                                       Condition_Type=ConditionType.expression,
                                       Condition_Value="/* Fallback: model returned no JSON */", DType="String")]]
                )
            else:
                try:
                    obj = json.loads(txt)
                    parsed = RuleMapResponse(**obj)
                except Exception:
                    heur = _heuristic_rulemap_from_text(text, columns_in_schema)
                    parsed = heur if heur else RuleMapResponse(
                        rule_name="", rule_details="", inputs=[],
                        groups=[[Statement(Column="", Operator=Operator.is_,
                                           Condition_Type=ConditionType.expression,
                                           Condition_Value="/* Fallback after parse error */", DType="String")]]
                    )
        else:
            parsed: RuleMapResponse = resp.parsed

        # --- Post-normalization ---
        normalized_groups: List[List[Statement]] = []
        for group in parsed.groups:
            norm_group: List[Statement] = []
            for stt in group:
                op_norm = _norm_op(stt.Operator.value if isinstance(stt.Operator, Operator) else str(stt.Operator))
                if op_norm == "is":                    op_enum = Operator.is_
                elif op_norm == "is not":              op_enum = Operator.is_not
                elif op_norm == "is within":           op_enum = Operator.is_within
                elif op_norm == "is not within":       op_enum = Operator.is_not_within
                elif op_norm == "is less than":        op_enum = Operator.is_less_than
                elif op_norm == "is less than or equal to": op_enum = Operator.is_less_equal
                elif op_norm == "is greater than":     op_enum = Operator.is_greater_than
                elif op_norm == "is greater than or equal to": op_enum = Operator.is_greater_equal
                elif op_norm == "contains":            op_enum = Operator.contains
                else:                                  op_enum = Operator.is_

                col_name = (stt.Column or "").strip()
                cond_val = (stt.Condition_Value or "").strip()

                # 'between' -> two statements
                if re.search(r"\bbetween\b", cond_val, flags=re.I):
                    m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)", cond_val, flags=re.I)
                    if m:
                        a, b = m.group(1).strip(), m.group(2).strip()
                        hint_a, hint_b = _infer_scalar_dtype_hint(a), _infer_scalar_dtype_hint(b)
                        ct_a = _infer_numeric_type(a) or ("expression" if _is_iso_date(a) else "string value")
                        ct_b = _infer_numeric_type(b) or ("expression" if _is_iso_date(b) else "string value")
                        norm_group += [
                            Statement(Column=col_name, Operator=Operator.is_greater_equal,
                                      Condition_Type=ConditionType(_norm_condition_type(ct_a)),
                                      Condition_Value=a, DType=hint_a),
                            Statement(Column=col_name, Operator=Operator.is_less_equal,
                                      Condition_Type=ConditionType(_norm_condition_type(ct_b)),
                                      Condition_Value=b, DType=hint_b)
                        ]
                        continue

                # within lists
                if op_enum in (Operator.is_within, Operator.is_not_within):
                    items = _split_items_natural(cond_val) if cond_val else []
                    dtype_hint = _infer_list_dtype_hint(items) if items else "String"
                    norm_group.append(Statement(Column=col_name, Operator=op_enum,
                                                Condition_Type=ConditionType.function,
                                                Condition_Value=", ".join(items), DType=dtype_hint))
                    continue

                # contains
                if op_enum == Operator.contains:
                    norm_group.append(Statement(Column=col_name, Operator=op_enum,
                                                Condition_Type=ConditionType.string_value,
                                                Condition_Value=cond_val, DType="String"))
                    continue

                # numeric/date comparisons
                if op_enum in (Operator.is_less_than, Operator.is_less_equal,
                               Operator.is_greater_than, Operator.is_greater_equal):
                    dtype_hint = _infer_scalar_dtype_hint(cond_val)
                    num_ct = _infer_numeric_type(cond_val)
                    ct_enum = ConditionType(_norm_condition_type(num_ct)) if num_ct \
                              else (ConditionType.expression if _is_iso_date(cond_val) else ConditionType.string_value)
                    norm_group.append(Statement(Column=col_name, Operator=op_enum,
                                                Condition_Type=ct_enum,
                                                Condition_Value=cond_val, DType=dtype_hint))
                    continue

                # is / is not
                if op_enum in (Operator.is_, Operator.is_not):
                    lower_cv = cond_val.lower()
                    if lower_cv in ("null", "none", ""):
                        ct_enum, cv, hint = ConditionType.null_value, "", None
                    elif lower_cv in ("current timestamp", "now", "today"):
                        ct_enum, cv, hint = ConditionType.current_timestamp, "", "Date/Time"
                    else:
                        hint = _infer_scalar_dtype_hint(cond_val)
                        num_ct = _infer_numeric_type(cond_val)
                        ct_enum = ConditionType(_norm_condition_type(num_ct)) if num_ct \
                                  else (ConditionType.expression if _is_iso_date(cond_val) else ConditionType.string_value)
                        cv = cond_val
                    norm_group.append(Statement(Column=col_name, Operator=op_enum,
                                                Condition_Type=ct_enum, Condition_Value=cv, DType=hint))
                    continue

                norm_group.append(stt)

            normalized_groups.append(norm_group)

        # --- Trim/strip hardener (regex fixed) ---
        if re.search(r"\b(trim|strip|leading|trailing\s+spaces)\b", text, flags=re.IGNORECASE):
            for g_idx, g in enumerate(normalized_groups):
                for s_idx, s in enumerate(g):
                    if (s.DType or "String") != "String":  continue
                    col = (s.Column or "").strip()
                    if not col:                             continue

                    if s.Operator in (Operator.is_within, Operator.is_not_within) and (s.Condition_Value or "").strip():
                        items = _split_items_natural(s.Condition_Value)
                        expr_items = ", ".join([f"'{v.upper()}'" for v in items]) if items else ""
                        normalized_groups[g_idx][s_idx] = Statement(
                            Column=col, Operator=Operator.is_,
                            Condition_Type=ConditionType.expression,
                            Condition_Value=f"Upper(Trim({col})) IN ({expr_items})",
                            DType="String"
                        )
                        continue

                    if s.Operator in (Operator.is_, Operator.is_not) and s.Condition_Type in (ConditionType.string_value, ConditionType.expression):
                        cv = (s.Condition_Value or "").strip()
                        if cv and cv.lower() not in ("null", "none", "current timestamp", "now", "today"):
                            normalized_groups[g_idx][s_idx] = Statement(
                                Column=col, Operator=Operator.is_,
                                Condition_Type=ConditionType.expression,
                                Condition_Value=f"Upper(Trim({col})) = '{cv.upper()}'",
                                DType="String"
                            )
                            continue

        # --- Completer: add uniqueness statements if NL text requires it ---
        uniq_needles = re.findall(r"['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\b[^.\n]*\b(unique|no\s+duplicates|distinct)\b", text, flags=re.IGNORECASE)
        uniq_cols_from_text = {m[0] for m in uniq_needles}
        if uniq_cols_from_text:
            existing_uniqs = {s.Column.strip() for g in normalized_groups for s in g
                              if s.Condition_Type == ConditionType.expression and (s.Condition_Value or "").startswith("IsUnique(")}
            for col in uniq_cols_from_text:
                if col and col not in existing_uniqs:
                    normalized_groups.append([
                        Statement(Column=col, Operator=Operator.is_,
                                  Condition_Type=ConditionType.expression,
                                  Condition_Value=f"IsUnique({col})",    # <-- Template uses IsUnique()
                                  DType="String")
                    ])

        # --- Build inputs ---
        seen: Dict[str, InputColumn] = {}
        rank = {"Date/Time": 3, "Float": 2, "Integer": 2, "String": 1}
        for group in normalized_groups:
            for s in group:
                col = (s.Column or "").strip()
                if not col: continue
                hint = s.DType or "String"
                dtype = _to_ui_dtype(hint)
                cur = seen.get(col)
                if not cur:
                    seen[col] = InputColumn(name=col, data_type=dtype, description="", max_length="")
                elif rank.get(dtype, 1) > rank.get(cur.data_type, 1):
                    cur.data_type = dtype
        inputs = list(seen.values())

        # --- Predict dimension via LLM + heuristic fallback ---
        try:
            class DimChoice(BaseModel):
                dimension: str  # one of ['Completeness','Uniqueness','Consistency','Accuracy','Timeliness','Integrity']

            dim_prompt = (
                "Given the following rule statements, classify the primary Data Quality dimension "
                "as one of: Completeness, Uniqueness, Consistency, Accuracy, Timeliness, Integrity.\n\n"
                "Statements:\n" +
                "\n".join([f"- {s.Column} {s.Operator} {s.Condition_Type} {s.Condition_Value or ''}"
                           for g in normalized_groups for s in g]) +
                "\n\nReturn ONLY JSON with field 'dimension'."
            )
            dim_resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=dim_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=DimChoice,
                                                   temperature=0.0, max_output_tokens=64)
            )
            predicted_dim = getattr(dim_resp, "parsed", None).dimension if getattr(dim_resp, "parsed", None) else None
        except Exception:
            predicted_dim = None

        if not predicted_dim:
            txt_all = (text or "").lower() + " " + " ".join([
                f"{s.Column} {s.Operator} {s.Condition_Type} {s.Condition_Value or ''}".lower()
                for g in normalized_groups for s in g
            ])
            if any(k in txt_all for k in ["null", "missing", "blank", "empty", "completeness"]):
                predicted_dim = "Completeness"
            elif any(k in txt_all for k in ["duplicate", "uniq", "unique", "distinct"]):
                predicted_dim = "Uniqueness"
            elif any(k in txt_all for k in ["format", "regex", "pattern", "length", "datatype", "type check", "valid"]):
                predicted_dim = "Consistency"
            elif any(k in txt_all for k in [">=", "<=", ">", "<", "between", "range", "within", "threshold", "accuracy"]):
                predicted_dim = "Accuracy"
            elif any(k in txt_all for k in ["timestamp", "time", "date", "recent", "late", "timeliness"]):
                predicted_dim = "Timeliness"
            elif any(k in txt_all for k in ["reference", "referential", "foreign key", "fk", "parent", "child", "integrity"]):
                predicted_dim = "Integrity"
            else:
                predicted_dim = "Consistency"

        out = RuleMapResponse(rule_name=parsed.rule_name, rule_details=parsed.rule_details, inputs=inputs, groups=normalized_groups, dimension=predicted_dim)
        out_dict = jsonable_encoder(out, by_alias=True, exclude_none=True)
        out_dict["dimension"] = predicted_dim
        return out_dict

    except Exception as e:
        try:
            print("[/nlp_rule_map] Exception:", e)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"NLP mapping failed: {e}")
