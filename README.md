# HR Policy Intelligence

FastAPI + plain JavaScript + local ChromaDB document store. Rebuilt from the provided UI reference.

## Answer flow

1. PDF/TXT uploads and indexing run locally, without Google or Groq.
2. A local extractive answer appears first, preserving source conditions and separating manuals.
3. With **Use AI to understand and answer the question** checked, the selected provider automatically interprets the original natural-language question using policy evidence.
4. For small manuals, the AI receives all pages of the selected document(s), regardless of keyword or topic matching. This supports paraphrases without maintaining an exhaustive question list.
5. For collections exceeding 18,000 text characters or 30 pages, generic keyword ranking and neighbouring pages select context. Coverage is marked incomplete; this is not a multilingual semantic index for large collections.
6. Answers cite short references tied to filename and page. Citation validation checks reference IDs, not factual entailment. Missing or conflicting information must be stated rather than inferred.
7. Successful AI answers are cached in memory for 10 minutes. Failures trigger a 30-second cooldown. Local results remain available during provider outages.

## Local run

Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://localhost:8000/#assistant. Use Overview to load synthetic sample policies.

## Render

Use `render.yaml`, or configure a Python Web Service:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health check: `/api/health`
- `DATA_DIR=/tmp/hr-policy-data` for ephemeral free-plan storage.

Free Render instances can sleep and lose uploaded data after restarts/deployments. For durable uploads, use a plan with a persistent disk and set DATA_DIR to a directory on that disk. Use one instance for embedded ChromaDB.

## Groq (optional alternative AI provider)

Create a key in https://console.groq.com/keys and set these secrets/variables in Render:

```text
CHAT_PROVIDER=groq
GROQ_API_KEY=<enter the key privately in Render>
GROQ_MODEL=openai/gpt-oss-20b
```

The Groq adapter uses https://api.groq.com/openai/v1/chat/completions. The model ID must be active and permitted for your account. Groq free plans have rate and token quotas; availability and answer quality are not guaranteed. Model access and plan terms can change.

## Gemini (optional)

```text
CHAT_PROVIDER=gemini
GEMINI_API_KEY=<enter the key privately in Render>
GEMINI_MODEL=<active model permitted for your account>
```

Selected policy text is sent only to the explicitly selected AI provider when AI answering is requested. Providers are not silently switched. A missing key leaves local answers available. Disabling AI in the UI sends no policy text to external AI services. Store all keys only in environment variables, never GitHub or the browser.

## Limits and verification

- PDF selectable text or UTF-8 TXT, max 10 MB. Scans need OCR, which is not included.
- No authentication, per-user document access, or LINE webhook yet. Use synthetic or approved public documents for public deployments.
- Local topic extraction is a fallback; it does not provide universal natural-language understanding. General interpretation requires the AI provider.
- Genuine free-provider uptime cannot be guaranteed. This app is designed to preserve local evidence when AI is unavailable.
- Optional provider adapters are tested using mocked HTTP responses. Production keys and actual model quality must be verified in your account.
- Run `pytest -q` for regression tests. Source policy PDFs are not committed to the repository.


## English questions over Thai documents

English questions are detected automatically. AI receives the original English question and responds in English while preserving the Thai policy's numbers, entitlement limits, paid/unpaid distinctions and conditions. Filename/page references and source passages remain in their original language.

For a Thai collection too large to send in full, the selected provider first translates the search query into Thai, then retrieval uses both the English question and Thai query. The final AI answer is generated in English from the selected original evidence. Small manuals are supplied in full and do not require a separate query-translation call. Query translation and answer caches reduce repeated calls. API callers can set `answer_language` to `auto`, `en`, or `th`.

If AI is unavailable, original policy excerpts are explicitly labelled as untranslated; local keyword search alone is not presented as a translation service. Source documents are never overwritten by translations.

## Reviewed policy evaluation

Evaluation runs eight local-answer checks per reviewed version of Manual_Dugong.pdf and Manual_Thaptaewan.pdf. Baselines include entitlement conditions, filename, source page and answer-reference linkage. Normalized content fingerprints detect changed versions. Unknown files are marked NOT_CONFIGURED; changed versions are VERSION_MISMATCH rather than FAIL. No AI calls are made by the evaluation endpoint. Passing these checks does not certify live AI translation quality, general answer accuracy or legal validity.
