import os
import json
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi

HF_USERNAME = "nduka1999"

class TriagePipeline:
    def __init__(self):
        # We start cold. We do NOT download massive models during module import
        # because HF Spaces will kill the container if the web server doesn't launch
        # within 3 minutes.
        self.is_loaded = False

        self.roberta_tokenizer = None
        self.roberta_model = None
        self.modernbert_tokenizer = None
        self.modernbert_model = None
        self.stacker = None
        self.knowledge_base = None
        self.bandit_state = None
        # Pre-built BM25 indices for RAG retrieval, keyed by product_id
        self.bm25_by_product = {}
        self.passages_by_product = {}

        # Exact Label Map from Target Variable Preprocessor
        self.label_map = {
            0: "Checking or savings account",
            1: "Credit card",
            2: "Credit reporting or other personal consumer reports",
            3: "Debt collection",
            4: "Debt or credit management",
            5: "Money transfer, virtual currency, or money service",
            6: "Mortgage",
            7: "Payday loan, title loan, personal loan, or advance loan",
            8: "Student loan",
            9: "Vehicle loan or lease"
        }

    def load_models(self):
        if self.is_loaded:
            return

        print("Lazy Loading Triage Pipeline and downloading artifacts from Hub...")

        # 1. Automatically Download & Cache Distilled RoBERTa
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/cfpb-roberta-distilled")
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(f"{HF_USERNAME}/cfpb-roberta-distilled")
        self.roberta_model.eval() # Set to evaluation mode

        # 2. Automatically Download & Cache ModernBERT
        self.modernbert_tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/cfpb-modernbert")
        self.modernbert_model = AutoModelForSequenceClassification.from_pretrained(f"{HF_USERNAME}/cfpb-modernbert")
        self.modernbert_model.eval()

        # 3. Download Ensemble Artifacts
        stacker_path = hf_hub_download(repo_id=f"{HF_USERNAME}/cfpb-ensemble-artifacts", filename="lr_stacker.joblib")
        kb_path = hf_hub_download(repo_id=f"{HF_USERNAME}/cfpb-ensemble-artifacts", filename="knowledge_base.json")
        bandit_path = hf_hub_download(repo_id=f"{HF_USERNAME}/cfpb-ensemble-artifacts", filename="bandit_state.json")

        self.stacker = joblib.load(stacker_path)

        with open(kb_path, "r") as f:
            self.knowledge_base = json.load(f)

        with open(bandit_path, "r") as f:
            self.bandit_state = json.load(f)

        # 4. Pre-build BM25 indices per product category for RAG retrieval
        # BM25-only approach: fast to build, no extra model download, lightweight on memory.
        # NB10 used hybrid BM25+MiniLM, but BM25-only is the deployment-safe fallback
        # to avoid the ~60s MiniLM encoding overhead on free-tier cold starts.
        print("Pre-computing BM25 indices per product category...")
        for passage in self.knowledge_base:
            pid = passage["product_id"]
            if pid not in self.passages_by_product:
                self.passages_by_product[pid] = []
            self.passages_by_product[pid].append(passage)

        for pid, passages in self.passages_by_product.items():
            tokenized = [p["text"].lower().split() for p in passages]
            self.bm25_by_product[pid] = BM25Okapi(tokenized)

        self.is_loaded = True
        print("All models and RAG indices fully loaded!")

    def is_ready(self):
        return self.is_loaded

    def route_with_bandit(self, pred_class_id, confidence):
        """Thompson Sampling routing using trained Beta-Bernoulli posteriors from NB11.

        4-arm contextual bandit (context = predicted product class 0-9):
          Arm 0: RoBERTa-D direct prediction
          Arm 1: ModernBERT direct prediction
          Arm 2: Ensemble (LR Stacker) — generally the strongest arm
          Arm 3: Human Escalation — fixed reward 0.5

        If confidence < 0.55 (NB11 Experiment 2 optimal threshold), force escalation.
        Otherwise, sample from Beta(alpha, beta) posteriors and pick the best arm.
        """
        arm_names = {
            0: "RoBERTa-D Direct",
            1: "ModernBERT Direct",
            2: "Ensemble (Stacked)",
            3: "Human Escalation"
        }

        # Low confidence → always escalate (NB11 threshold sweep result)
        if confidence < 0.55:
            return {
                "decision": arm_names[3],
                "arm": 3,
                "reason": f"Confidence {confidence:.3f} below 0.55 threshold — forced escalation"
            }

        # Sample from Beta posteriors for this product context
        # bandit_state["alpha"] and ["beta"] are both [10 contexts][4 arms]
        context = int(pred_class_id)
        samples = [
            np.random.beta(
                self.bandit_state["alpha"][context][a],
                self.bandit_state["beta"][context][a]
            )
            for a in range(4)
        ]
        selected_arm = int(np.argmax(samples))

        return {
            "decision": arm_names[selected_arm],
            "arm": selected_arm,
            "reason": f"Thompson samples: [{', '.join(f'{s:.3f}' for s in samples)}]"
        }

    def retrieve_context(self, text, pred_class_id, top_k=3):
        """BM25 sparse retrieval over CFPB regulatory knowledge base.

        Deployment-optimised version of the NB10 hybrid pipeline:
        1. Pre-filter knowledge_base by predicted product_id (2,689 → ~200-300)
        2. BM25 sparse scoring on pre-built index
        3. Return top_k passages with source metadata

        The full hybrid (BM25 + MiniLM dense + RRF) from NB10 can be restored
        by adding sentence-transformers back once cold start time is acceptable.
        """
        pid = int(pred_class_id)

        # If no passages for this product, return empty
        if pid not in self.passages_by_product:
            return []

        passages = self.passages_by_product[pid]
        bm25 = self.bm25_by_product[pid]

        # BM25 sparse scores
        bm25_scores = bm25.get_scores(text.lower().split())

        # Walk ranked results, skip passages whose truncated text we've already seen.
        # This prevents near-duplicate regulatory passages from filling all 3 slots.
        ranked_indices = np.argsort(-bm25_scores)
        results = []
        seen_texts = set()
        for i in ranked_indices:
            snippet = passages[i]["text"][:500]
            if snippet in seen_texts:
                continue
            seen_texts.add(snippet)
            results.append({
                "text": snippet,
                "source": passages[i]["source_doc"],
                "issue": passages[i]["issue_name"]
            })
            if len(results) >= top_k:
                break

        return results

    def process(self, text):
        """Processes the text through the full agentic pipeline and formats the output for the frontend."""
        # Ensure models are loaded before processing the text
        if not self.is_loaded:
            self.load_models()

        trace = ["1. Models loaded from HF Hub (cached)."]

        # 1. Inference via RoBERTa-D
        # max_length=384 matches NB08 training config (NOT 512)
        roberta_inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=384)
        with torch.no_grad():
            roberta_outputs = self.roberta_model(**roberta_inputs)
            roberta_probs = torch.nn.functional.softmax(roberta_outputs.logits, dim=-1).numpy()

        # Log RoBERTa-D's own top prediction for the trace
        roberta_pred_id = int(np.argmax(roberta_probs[0]))
        roberta_conf = float(roberta_probs[0][roberta_pred_id])
        trace.append(f"2. RoBERTa-D inference: {self.label_map[roberta_pred_id]} ({roberta_conf:.2f})")

        # 2. Inference via ModernBERT
        # max_length=384 matches NB06b training config (NOT 512)
        modernbert_inputs = self.modernbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=384)
        with torch.no_grad():
            modernbert_outputs = self.modernbert_model(**modernbert_inputs)
            modernbert_probs = torch.nn.functional.softmax(modernbert_outputs.logits, dim=-1).numpy()

        # Log ModernBERT's own top prediction for the trace
        modernbert_pred_id = int(np.argmax(modernbert_probs[0]))
        modernbert_conf = float(modernbert_probs[0][modernbert_pred_id])
        trace.append(f"3. ModernBERT inference: {self.label_map[modernbert_pred_id]} ({modernbert_conf:.2f})")

        # 3. Stack via Logistic Regression Classifier (StandardScaler + LR pipeline from NB13)
        X_stack = np.hstack([roberta_probs, modernbert_probs])
        pred_class_id = self.stacker.predict(X_stack)[0]
        confidence = float(np.max(self.stacker.predict_proba(X_stack)[0]))

        predicted_label = self.label_map.get(pred_class_id, f"Class {pred_class_id}")
        trace.append(f"4. LR Stacker decision: {predicted_label} ({confidence:.2f})")

        # 4. Thompson Sampling bandit routing (NB11)
        routing = self.route_with_bandit(pred_class_id, confidence)
        trace.append(f"5. Thompson Sampling: Arm {routing['arm']} ({routing['decision']}) selected")

        # 5. RAG retrieval — BM25 over CFPB regulatory passages (NB10)
        rag_passages = self.retrieve_context(text, pred_class_id, top_k=3)
        trace.append(f"6. RAG retrieved {len(rag_passages)} passages for '{predicted_label}'")

        # Return neatly formatted payload for the Vercel React frontend
        return {
            "summary": f"Classified as: {predicted_label} ({confidence*100:.1f}%)",
            "classification": {
                "label": predicted_label,
                "confidence": confidence
            },
            "routing": routing,
            "rag_context": rag_passages,
            "agentic_trace": trace
        }
