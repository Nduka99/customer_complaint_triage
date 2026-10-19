import { Client } from "@gradio/client";

// The Space ID comes from .env so it's easy to change without touching code.
// Vite replaces "import.meta.env.VITE_SPACE_ID" with the actual value at build time.
const SPACE_ID = import.meta.env.VITE_SPACE_ID || "nduka1999/cfpb-triage-backend";

// We keep one shared client instance so we don't reconnect on every request.
let clientInstance = null;

// The Space stages the Hugging Face Hub reports while a Space wakes, grouped into the
// steps the app shows: 0 asleep, 1 starting (models and search index load), 2 running.
const WAKE_STEPS = {
  SLEEPING: 0,
  STOPPED: 0,
  BUILDING: 1,
  APP_STARTING: 1,
  RUNNING_BUILDING: 1,
  RUNNING: 2,
};

/**
 * Ask the Hub how far the Space is in waking up.
 * Returns 0, 1 or 2 (see WAKE_STEPS), or null when the Hub can't be reached or the
 * Space reports a problem (paused, runtime error), so waiting longer won't help.
 */
export async function getWakeStep() {
  try {
    const response = await fetch(`https://huggingface.co/api/spaces/${SPACE_ID}`);
    if (!response.ok) return null;
    const space = await response.json();
    return WAKE_STEPS[space.runtime?.stage] ?? null;
  } catch {
    return null;
  }
}

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
