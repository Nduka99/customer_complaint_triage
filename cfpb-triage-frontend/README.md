# Agentic Triage Frontend

React-based dashboard for the [Agentic Consumer Complaint Triage System](https://cfpb-triage-nduka.vercel.app). Provides a real-time interface for submitting consumer complaints and viewing classification results, regulatory context, and routing decisions from the ML pipeline.

## Live Demo

**URL**: [cfpb-triage-nduka.vercel.app](https://cfpb-triage-nduka.vercel.app)

## Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 19.2.4 | UI framework |
| Vite | 8.0.1 | Build tool and dev server |
| Tailwind CSS | 4.2.2 | Utility-first styling |
| @gradio/client | 2.1.0 | Communication with HF Spaces backend |
| ESLint | 9.x | Code quality |

## Features

- **Complaint input**: Textarea with pre-loaded example complaints for quick testing
- **Classification results**: Product category label with confidence score
- **Agentic trace**: Step-by-step pipeline execution trace (RoBERTa-D → ModernBERT → Ensemble → Bandit → RAG)
- **RAG context panel**: Retrieved CFPB regulatory examination passages with source attribution
- **Routing badge**: Thompson Sampling decision indicator (which arm was selected and why)
- **Cold-start handling**: Displays cached example results while the HF Spaces backend wakes from sleep, so users see a functional interface immediately

## Components

| Component | Purpose |
|-----------|---------|
| `App.jsx` | Root orchestrator, two-column dashboard layout, backend connection logic |
| `ComplaintInput.jsx` | Textarea with example complaint buttons |
| `ResultsDashboard.jsx` | Classification label and confidence display |
| `AgenticTrace.jsx` | Step-by-step pipeline trace visualisation |
| `RagContext.jsx` | Retrieved regulatory passage cards |
| `RoutingBadge.jsx` | Thompson Sampling routing decision indicator |
| `LoadingState.jsx` | Backend connection waiting screen |
| `api.js` | Gradio client integration (connect, classify, health check) |
| `cachedExample.js` | Pre-cached demo results for offline/cold-start preview |

## Deployment Details

- **Platform**: Vercel (Hobby tier — free)
- **Delivery**: Static SPA via Vercel's global edge CDN
- **Cost**: $0/month
- **Build output**: `dist/` directory (Vite-compiled HTML/JS/CSS)
- **Backend connection**: Communicates with `nduka1999/cfpb-triage-backend` on HF Spaces via the Gradio Client JS library

## Local Development

```bash
npm install
npm run dev
```

The dev server launches at `http://localhost:5173` with hot module replacement.

## Build

```bash
npm run build
```

Produces optimised static assets in `dist/` ready for deployment.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_SPACE_ID` | `nduka1999/cfpb-triage-backend` | HF Space ID for backend connection |
