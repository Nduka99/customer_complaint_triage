# Complaint Triage — Frontend

React dashboard for the Agentic Consumer Complaint Triage System. Provides a real-time interface for submitting consumer complaints and viewing classification results, routing decisions, and relevant regulation from the ML pipeline.

## Live Demo

**URL**: https://customer-complaint-triage.vercel.app

## Tech Stack

| Technology     | Version | Purpose                              |
| -------------- | ------- | ------------------------------------ |
| React          | 19.2.4  | UI framework                         |
| Vite           | 8.0.1   | Build tool and dev server            |
| Tailwind CSS   | 4.2.2   | Utility-first styling                |
| @gradio/client | 2.1.0   | Communication with HF Spaces backend |
| ESLint         | 9.x     | Code quality                         |

## Features

- **Complaint input**: Textarea with pre-loaded example complaints (clear-cut and deliberately ambiguous cases)
- **Classification results**: Product category label with a qualitative confidence meter
- **Decision trace**: Animated step-by-step explanation of how the decision was made, in plain language
- **Two-layer UX**: Plain product language by default; a "Show technical details" toggle reveals the raw pipeline internals (model names, Thompson samples, automation thresholds) for technical reviewers
- **Relevant regulation panel**: Retrieved CFPB examination-procedure excerpts with source attribution
- **Routing badge**: Automated handling vs. referral for human review, with a plain-language reason
- **Cold-start handling**: Friendly startup screen with progress stages and cached sample results while the HF Spaces backend wakes from sleep, plus a retry button on connection failure

## Components

| Component                | Purpose                                                                       |
| ------------------------ | ----------------------------------------------------------------------------- |
| `App.jsx`              | Root orchestrator, two-column layout, connection state, technical-details toggle |
| `ComplaintInput.jsx`   | Textarea with example complaint buttons                                       |
| `ResultsDashboard.jsx` | Category label and confidence meter                                           |
| `AgenticTrace.jsx`     | Parses raw backend trace lines into plain-language steps (raw log in tech layer) |
| `RagContext.jsx`       | Retrieved regulation excerpt cards with show more/less                        |
| `RoutingBadge.jsx`     | Routing outcome (automated vs. human review) with plain-language reason       |
| `LoadingState.jsx`     | Cold-start waiting screen with startup stages and retry                       |
| `api.js`               | Gradio client integration (connect, classify, health check)                   |
| `cachedExample.js`     | Pre-cached demo results for cold-start preview                                |

## Deployment Details

- **Platform**: Vercel (Hobby tier — free), git-integrated: pushes to `main` auto-deploy
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

| Variable          | Default                           | Purpose                            |
| ----------------- | --------------------------------- | ---------------------------------- |
| `VITE_SPACE_ID` | `nduka1999/cfpb-triage-backend` | HF Space ID for backend connection |
