# Human-in-the-Loop (HITL) AI Document Pipeline

A multi-step, retrieval-augmented document generation pipeline where **every AI output must be reviewed and approved by a human before the workflow can advance**.

A user provides an input document (a brief, a spec, a set of instructions). The system extracts its requirements, then walks through four AI-assisted stages, each one grounded in a knowledge base and gated by an explicit human decision. Every decision is recorded in an audit trail.

The demo runs fully offline with deterministic mock generators. The `app/ai/` layer contains the production integrations for Claude on Amazon Bedrock, llama.cpp embeddings, and a FAISS vector index.

## Pipeline

```text
Input Document
      │
      ▼
 Requirement Extraction ──► regex heuristics (+ optional LLM pass)
      │
      ▼
 1. Structured Outline  ──► HITL gate
      │
      ▼
 2. Key Highlights      ──► HITL gate
      │
      ▼
 3. Narrative Draft     ──► HITL gate
      │
      ▼
 4. Quality Validation  ──► HITL gate
      │
      ▼
   Complete (full audit trail)
```

At each gate the reviewer can:

- **Accept** the output (a reviewer ID is required and logged)
- **Edit inline**, then accept the edited version
- **Reject** with a written reason
- **Request an alternative** generation
- **Re-draft with notes**, giving the AI specific guidance to apply

A step stays locked until the previous step is approved. The AI never approves or advances anything on its own.

## Features

- **Retrieval-grounded output.** Each step retrieves knowledge-base sources and shows them as scored chips in the UI.
- **Persistent working document.** The outline, highlights, and draft build on one document instead of producing disconnected fragments.
- **Validation scoring.** Deterministic rules plus an optional LLM pass flag gaps by severity (critical, major, minor), produce a 0–100 estimate, and include a "Fix ▶" jump-to-location helper.
- **Audit trail.** Every accept, edit, reject, alternative, and re-draft is recorded with step, reviewer, timestamp, attempt number, and notes.
- **Feedback capture.** The `app/hitl/` layer tags reviewer feedback (hallucination, tone, missing content, sources) for later analysis of acceptance and edit rates.

## Tech stack

| Layer | Demo | Production path (`app/ai/`) |
|---|---|---|
| Backend | FastAPI | FastAPI |
| Frontend | Jinja2 + vanilla HTML/CSS/JS | same |
| Generation | Deterministic templates | Claude via Amazon Bedrock (optional LiteLLM proxy) |
| Retrieval | Keyword-overlap scoring over JSON | llama.cpp embeddings + FAISS |
| State | In-memory sessions | PostgreSQL |
| Audit | In-memory history | JSONL file + database |

## Project structure

```text
.
├── README.md
├── LICENSE
├── config.py                  # pydantic-settings configuration
├── run_demo.py                # local dev entry point
├── requirements.txt           # demo dependencies
├── requirements-ai.txt        # full AI stack dependencies
├── .env.example
├── app/
│   ├── main.py                # FastAPI routes + HITL gating
│   ├── models.py              # Pydantic models (session, steps, flags, audit records)
│   ├── workflow.py            # offline mock generators + requirement extraction
│   ├── knowledge_base.py      # lightweight keyword retrieval
│   ├── ai/
│   │   ├── bedrock_client.py      # Claude/Bedrock client + prompt builders
│   │   ├── document_parser.py     # two-pass requirement extraction
│   │   ├── embeddings.py          # llama.cpp embedding client + chunking
│   │   ├── faiss_index.py         # FAISS index build/load/search
│   │   ├── rag_pipeline.py        # retrieval → validation gate → generation
│   │   ├── reference_matcher.py   # case-study-to-requirement matching
│   │   └── validation_checker.py  # rule-based + AI gap analysis
│   ├── hitl/
│   │   ├── audit_trail.py         # append-only decision log
│   │   ├── feedback_capture.py    # reviewer feedback tagging + metrics
│   │   └── review_queue.py        # multi-reviewer claim/resolve queue
│   ├── static/
│   │   ├── app.js
│   │   └── styles.css
│   └── templates/
│       └── index.html
└── data/
    ├── input_document.txt     # sample input (placeholder text)
    └── knowledge_base.json    # sample KB entries (placeholder text)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_demo.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), click **Load Sample Document**, then **Upload & Analyze Document** to walk through the pipeline.

### Enabling the full AI stack

```bash
pip install -r requirements-ai.txt
cp .env.example .env             # add AWS credentials and service URLs
```

Then swap the mock functions in `app/workflow.py` for the equivalent `RAGPipeline` methods in `app/ai/rag_pipeline.py`.

## API overview

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/document/upload` | Create a session and extract requirements |
| `GET` | `/api/session/{id}` | Full session state |
| `POST` | `/api/step/{step}/generate` | Generate output for `outline`, `highlights`, or `draft` |
| `POST` | `/api/step/validation/run` | Run the validation check |
| `POST` | `/api/step/{step}/review` | Submit a HITL decision |
| `POST` | `/api/step/{step}/redraft` | Regenerate with reviewer notes |
| `GET` | `/api/session/{id}/audit` | Audit trail for a session |
| `GET` | `/api/sample-document` | Bundled sample input |

## Limitations

This is a demo, not a production system. It does not include authentication, authorization, encrypted storage, or persistent sessions. Add those before any real use.

## Design principle

**The AI drafts. People decide.**
