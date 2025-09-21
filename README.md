# DQ Relay for Gemini + Qdrant

A minimal FastAPI relay that:
- Upserts your DQ artifacts (profile report extracts, rule CSV text, DQ execution reports)
- Runs RAG with Qdrant + Gemini
- Exposes /chat to your Streamlit app

## Environment variables

- GEMINI_API_KEY: your Google Generative AI API key
- QDRANT_URL: Qdrant endpoint (Cloud or self-hosted)
- QDRANT_API_KEY: Qdrant API key
- QDRANT_COLLECTION: collection name (e.g., dq_docs)
- EMBED_MODEL: text-embedding-004 (default)
- GEN_MODEL: gemini-1.5-flash (default)
- CORS_ORIGINS: comma-separated origins (e.g., https://your-streamlit.company.com)
- AUTH_TOKEN: optional bearer token for relay auth

## Deploy on Render

1. Create a new Web Service from this repo.
2. Runtime: Python 3.11
3. Build command: `pip install -r requirements.txt`
4. Start command: leave empty (Procfile used)
5. Add Environment variables as above.
6. Deploy.

## Load your documents

Call /upsert_batch with your DQ assets chunked as text:

POST /upsert_batch
Authorization: Bearer $AUTH_TOKEN
Content-Type: application/json
