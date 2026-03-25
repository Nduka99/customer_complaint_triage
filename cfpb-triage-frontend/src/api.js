import { Client } from "@gradio/client";

// The Space ID comes from .env so it's easy to change without touching code.
// Vite replaces "import.meta.env.VITE_SPACE_ID" with the actual value at build time.
const SPACE_ID = import.meta.env.VITE_SPACE_ID || "nduka1999/cfpb-triage-backend";

// We keep one shared client instance so we don't reconnect on every request.
let clientInstance = null;

/**
 * Connect to the HF Space. Returns the Gradio client object.
 *
 * This is separated from classify() so the frontend can attempt connection
 * on page load and detect whether the backend is sleeping (cold start).
 */
export async function connectToBackend() {
  if (clientInstance) return clientInstance;
  clientInstance = await Client.connect(SPACE_ID);
  return clientInstance;
}

/**
 * Send a complaint to the backend and get the triage result.
 *
 * Uses the Gradio client's .predict() method, which calls the endpoint
 * we named "classify" in app.py (api_name="classify").
 *
 * The backend returns a single JSON object from pipeline.process():
 *   { summary, classification, routing, rag_context, agentic_trace }
 *
 * Gradio wraps that in result.data — the first element is our payload.
 */
export async function classifyComplaint(text) {
  const client = await connectToBackend();
  const result = await client.predict("/classify", { text });
  return result.data[0];
}

/**
 * Quick health check — just tries to connect.
 * Returns true if the backend is awake, false if it's sleeping/unreachable.
 */
export async function checkHealth() {
  try {
    await connectToBackend();
    return true;
  } catch {
    return false;
  }
}
