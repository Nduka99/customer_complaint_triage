import gradio as gr
from pipeline import TriagePipeline

# Initialise the pipeline (lazy-loads models on first request, not at import time)
pipeline = TriagePipeline()

def classify_complaint(text):
    """Main classification function — exposed via both the Gradio UI and the auto-generated API."""
    if not text or not text.strip():
        return {"error": "No complaint text provided."}
    return pipeline.process(text)

# Build the Gradio Interface
# Gradio auto-generates a REST endpoint at /api/predict (or /call/classify)
# that the React frontend can call — no separate FastAPI needed.
demo = gr.Interface(
    fn=classify_complaint,
    inputs=gr.Textbox(lines=8, label="Consumer Complaint", placeholder="Paste a consumer complaint here..."),
    outputs=gr.JSON(label="Triage Result"),
    title="CFPB Agentic Triage System",
    description="Paste a consumer complaint below to classify it through the ensemble pipeline.",
    api_name="classify",   # creates /api/classify endpoint automatically
    examples=[
        ["I opened a checking account and was charged hidden fees that were not disclosed during signup."],
        ["A debt collector keeps calling me about a debt I already paid off and is threatening legal action."],
        ["My credit report shows a late payment that I never made. I have proof of on-time payment."],
    ],
)

# launch() starts the Gradio server — this is what keeps the HF Space alive.
# HF Spaces requires binding to 0.0.0.0 to be reachable.
demo.launch(server_name="0.0.0.0", server_port=7860)
