---
title: Cfpb Triage Backend
emoji: 🌖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: mit
short_description: Backend inference engine for the Agentic Consumer Complaint Triage system
---

# Agentic Triage Backend

Inference engine for the [Agentic Consumer Complaint Triage System](https://cfpb-triage-nduka.vercel.app), deployed on Hugging Face Spaces. Processes consumer financial complaints through a multi-stage ML pipeline and returns classification, regulatory context, and routing decisions as a JSON payload.

## Live Endpoint

**Space**: [nduka1999/cfpb-triage-backend](https://huggingface.co/spaces/nduka1999/cfpb-triage-backend)
**API**: `POST /api/classify` — accepts complaint text, returns triage result

## Pipeline Architecture

```
Complaint text
      |
      v
[1] RoBERTa-D inference (125M params, max_length=384)
      |
      +-- confidence < 0.65 --> Early exit: force human escalation
      |
      v
[2] ModernBERT inference (149M params, max_length=384)
      |
      v
[3] LR Stacking Ensemble (combines probability distributions)
      |
      v
[4] Thompson Sampling Bandit (4-arm routing, escalation threshold 0.55)
      |
      v
[5] BM25 RAG retrieval (2,689 CFPB regulatory passages, product-filtered)
      |
      v
JSON response: { summary, classification, routing, rag_context, agentic_trace }
```

## Key Design Decisions

- **Lazy loading**: Models are downloaded from HF Hub on first request (not at import time) to meet the 3-minute boot deadline on HF Spaces.
- **Early-exit gate**: Complaints with RoBERTa-D confidence below 0.65 skip the full pipeline entirely, saving compute and forcing immediate human escalation for ambiguous inputs.
- **BM25-only retrieval**: The full hybrid BM25 + MiniLM dense retrieval from NB10 was simplified to BM25-only to avoid ~60s MiniLM encoding overhead on free-tier cold starts.
- **Pre-built BM25 indices**: One BM25 index per product category is built at load time, so retrieval at request time is a fast sparse lookup.

## Models (loaded from HF Hub)

| Artifact | Repository | Purpose |
|----------|-----------|---------|
| RoBERTa-D | `nduka1999/cfpb-roberta-distilled` | Primary classifier (knowledge-distilled, F1=0.750) |
| ModernBERT | `nduka1999/cfpb-modernbert` | Secondary classifier (F1=0.736) |
| LR Stacker | `nduka1999/cfpb-ensemble-artifacts` | Ensemble combiner (F1=0.757) |
| Knowledge Base | `nduka1999/cfpb-ensemble-artifacts` | 2,689 regulatory passages for RAG |
| Bandit State | `nduka1999/cfpb-ensemble-artifacts` | Beta posteriors for Thompson Sampling |

## Deployment Details

- **Platform**: Hugging Face Spaces (cpu-basic free tier — 16 GB RAM, 2 CPU, 50 GB disk)
- **Framework**: Gradio 6.9.0 with auto-generated REST API
- **Cost**: $0/month
- **Cold start**: ~30-60s on first request after idle period (model download + caching)
- **Warm latency**: ~2-3s per request (CPU inference)

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

The Gradio interface launches at `http://localhost:7860` with both a web UI and REST API.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio interface, API endpoint definition, example complaints |
| `pipeline.py` | `TriagePipeline` class: model loading, inference, ensemble, RAG, routing |
| `requirements.txt` | Python dependencies |

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
