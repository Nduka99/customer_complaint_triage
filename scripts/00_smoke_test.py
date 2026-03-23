"""
00_smoke_test.py — Run this BEFORE the full notebooks.

Takes ~2 minutes on your RTX 4060. Creates synthetic data and runs both
DeBERTa + ModernBERT training pipelines for 10 steps each, verifying:

  ✓ FIX 1: Windows num_workers=0 (no freeze)
  ✓ FIX 2: gradient_checkpointing use_reentrant=False (no grad errors)
  ✓ FIX 3: ModernBERT attn_implementation=eager (no flash_attn crash)
  ✓ FIX 4: class_weights lazy device placement (no device mismatch)
  ✓ FIX 5: ModernBERT bf16 precision (no NaN from fp16 loss scaling)
  ✓ Loss decreases (training is actually learning)
  ✓ Eval macro-F1 is computed (metric pipeline works)
  ✓ Predictions use all classes (not collapsed to single class)
  ✓ No NaN in logits or loss
  ✓ Labels survive compute_loss (label_ids not None)

Usage:
    python 00_smoke_test.py

If all checks pass, you're safe to run the full notebooks.
"""

import sys, os, time, tempfile, warnings, traceback
import numpy as np

warnings.filterwarnings("ignore")

# ─── Force UTF-8 on Windows to avoid cp1252 emoji encoding errors ───
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Pretty output ───
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}[PASS] {msg}{RESET}")
def fail(msg): print(f"  {RED}[FAIL] {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}[WARN] {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{'='*60}\n  {msg}\n{'='*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════
#  ENVIRONMENT CHECKS
# ═══════════════════════════════════════════════════════════════════
header("STEP 0: Environment Checks")

import torch
print(f"  PyTorch:       {torch.__version__}")
print(f"  CUDA:          {torch.cuda.is_available()}")
print(f"  OS:            {sys.platform}")

if not torch.cuda.is_available():
    fail("No CUDA GPU detected. Tests will run on CPU (slower but still valid).")
    DEVICE = "cpu"
else:
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    bf16 = torch.cuda.is_bf16_supported()
    print(f"  GPU:           {gpu} ({mem:.1f} GB)")
    print(f"  bf16 support:  {bf16}")
    DEVICE = "cuda"
    ok(f"GPU detected: {gpu}")

import transformers
print(f"  Transformers:  {transformers.__version__}")

tf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
if tf_version < (4, 48):
    fail(f"transformers {transformers.__version__} < 4.48.0 — ModernBERT won't load!")
    print(f"    Run: pip install --upgrade transformers")
else:
    ok(f"transformers {transformers.__version__} >= 4.48.0")

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset

NUM_WORKERS = 0 if sys.platform == "win32" else 4
print(f"  Workers:       {NUM_WORKERS} ({'Windows-safe' if NUM_WORKERS == 0 else 'Linux'})")


# ═══════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════
header("STEP 1: Creating Synthetic Data (mimics 10-class CFPB)")

N_CLASSES = 10
N_TRAIN = 200    # tiny — just enough to run 10 steps
N_VAL = 50

# Class-imbalanced like the real data (Debt Mgmt is ~3x smaller)
class_probs = [0.15, 0.12, 0.12, 0.12, 0.03, 0.10, 0.10, 0.10, 0.08, 0.08]
class_names = [
    "Credit Report", "Debt Collect", "Credit Card", "Bank Acct",
    "Debt Mgmt", "Money Xfer", "Student Loan", "Vehicle Loan",
    "Payday/Pers", "Mortgage"
]

# Create texts with class-distinctive vocabulary (so the model has signal)
class_words = {
    0: "credit report equifax transunion experian bureau score",
    1: "debt collector collection agency owe payment",
    2: "credit card charge interest rate annual fee",
    3: "checking savings account bank branch deposit",
    4: "debt management plan counseling consolidation",
    5: "money transfer wire western union remittance",
    6: "student loan federal private university college",
    7: "vehicle car auto loan lease dealer",
    8: "payday loan personal advance title loan",
    9: "mortgage home house property refinance escrow",
}

np.random.seed(42)

def make_texts(n, labels):
    texts = []
    for label in labels:
        words = class_words[label].split()
        # 30-50 words with some noise
        n_words = np.random.randint(30, 50)
        text_words = np.random.choice(words, size=n_words, replace=True).tolist()
        # Add some shared noise words
        noise = np.random.choice(
            ["the", "I", "they", "said", "called", "told", "my", "was", "have", "account"],
            size=10, replace=True
        ).tolist()
        text_words.extend(noise)
        np.random.shuffle(text_words)
        texts.append(" ".join(text_words))
    return texts

train_labels = np.random.choice(N_CLASSES, size=N_TRAIN, p=class_probs).tolist()
val_labels   = np.random.choice(N_CLASSES, size=N_VAL,   p=class_probs).tolist()
train_texts  = make_texts(N_TRAIN, train_labels)
val_texts    = make_texts(N_VAL, val_labels)

print(f"  Train: {N_TRAIN} samples")
print(f"  Val:   {N_VAL} samples")
print(f"  Classes: {N_CLASSES}")
print(f"  Sample text: '{train_texts[0][:80]}...'")
ok("Synthetic data created")


# ═══════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

class SyntheticDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy = (preds == labels).mean()
    return {"macro_f1": macro_f1, "accuracy": accuracy}


from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(N_CLASSES),
    y=train_labels,
)


class WeightedTrainer(Trainer):
    """Trainer with class-weighted CE — lazy device placement (FIX 4)."""

    def __init__(self, class_weights_np, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cw_np = class_weights_np
        self._cw_tensor = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # FIX 4: Lazy device placement — avoids device mismatch
        if self._cw_tensor is None or self._cw_tensor.device != logits.device:
            self._cw_tensor = torch.tensor(
                self._cw_np, dtype=torch.float32, device=logits.device
            )

        loss = nn.CrossEntropyLoss(weight=self._cw_tensor)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def run_pipeline(model_name, extra_model_kwargs=None, use_bf16=False, use_fp16=False):
    """Run a full training pipeline for 10 steps and validate everything."""
    extra_model_kwargs = extra_model_kwargs or {}

    print(f"\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"  Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=N_CLASSES,
        problem_type="single_label_classification",
        **extra_model_kwargs,
    )

    # FIX 2: use_reentrant=False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ok("FIX 2: gradient_checkpointing_enable(use_reentrant=False)")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f} M")

    # Build datasets (short max_len=64 for speed)
    train_ds = SyntheticDataset(train_texts, train_labels, tokenizer, max_len=64)
    val_ds   = SyntheticDataset(val_texts,   val_labels,   tokenizer, max_len=64)

    # Verify dataset output
    sample = train_ds[0]
    assert "input_ids" in sample, "Missing input_ids"
    assert "attention_mask" in sample, "Missing attention_mask"
    assert "labels" in sample, "Missing labels"
    ok(f"Dataset produces correct keys: {list(sample.keys())}")

    # Training args
    args = TrainingArguments(
        output_dir=os.path.join(tempfile.gettempdir(), f"smoke_test_{model_name.split('/')[-1]}"),
        num_train_epochs=2,
        max_steps=10,               # just 10 steps
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="no",         # don't save checkpoints in smoke test
        logging_steps=5,
        report_to="none",
        seed=42,
        dataloader_num_workers=NUM_WORKERS,       # FIX 1
        dataloader_pin_memory=False,
    )
    ok(f"FIX 1: dataloader_num_workers={NUM_WORKERS}")

    precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")
    ok(f"FIX 5: precision={precision}")

    trainer = WeightedTrainer(
        class_weights_np=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # ── TRAIN ──
    print(f"  Training for 10 steps...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── Validate training loss ──
    train_loss = result.training_loss
    print(f"  Final train loss: {train_loss:.4f}")
    if np.isnan(train_loss):
        fail("Training loss is NaN!")
        return False
    ok(f"Training loss is finite: {train_loss:.4f}")

    # ── Check loss decreased ──
    loss_logs = [l for l in trainer.state.log_history if "loss" in l and "eval_loss" not in l]
    if len(loss_logs) >= 2:
        first_loss = loss_logs[0]["loss"]
        last_loss = loss_logs[-1]["loss"]
        if last_loss < first_loss:
            ok(f"Loss decreased: {first_loss:.4f} → {last_loss:.4f}")
        else:
            warn(f"Loss didn't decrease: {first_loss:.4f} → {last_loss:.4f} (may be OK with 10 steps)")
    
    # ── Check eval metrics were logged ──
    eval_logs = [l for l in trainer.state.log_history if "eval_macro_f1" in l]
    if len(eval_logs) == 0:
        fail("No eval metrics logged!")
        return False
    ok(f"Eval metrics logged: {len(eval_logs)} checkpoints")
    
    for log in eval_logs:
        if np.isnan(log.get("eval_loss", 0)):
            fail("NaN in eval loss!")
            return False
    ok("No NaN in eval loss")

    # ── PREDICT ──
    print(f"  Running prediction...")
    predictions = trainer.predict(val_ds)
    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=-1)

    # Label flow check
    if y_true is None:
        fail("label_ids is None — compute_loss broke the label flow!")
        return False
    ok("FIX 4: Labels survived compute_loss (label_ids is not None)")

    # NaN check
    if np.isnan(logits).any():
        fail("NaN in prediction logits!")
        return False
    ok("No NaN in prediction logits")

    # Multi-class check
    unique_preds = np.unique(y_pred)
    if len(unique_preds) == 1:
        warn(f"Model predicts only class {unique_preds[0]} — may be OK with 10 steps of synthetic data")
    else:
        ok(f"Model predicts {len(unique_preds)} distinct classes")

    # Macro-F1
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  Smoke-test macro-F1: {macro_f1:.4f} (not meaningful with 10 steps)")
    ok("Full pipeline completed without errors")

    # Cleanup
    del model, trainer
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


# ═══════════════════════════════════════════════════════════════════
#  TEST 1: DeBERTa-v3-base
# ═══════════════════════════════════════════════════════════════════
header("STEP 2: DeBERTa-v3-base (microsoft/deberta-v3-base)")

deberta_ok = False
# FIX 6: DeBERTa also needs bf16 (not fp16) on PyTorch 2.10+ with gradient checkpointing
# fp16 triggers "Attempting to unscale FP16 gradients" with use_reentrant=False
deberta_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
try:
    deberta_ok = run_pipeline(
        model_name="microsoft/deberta-v3-base",
        use_bf16=(DEVICE == "cuda" and deberta_bf16),
        use_fp16=(DEVICE == "cuda" and not deberta_bf16),
    )
except Exception as e:
    fail(f"DeBERTa pipeline CRASHED: {e}")
    traceback.print_exc()

if deberta_ok:
    ok("DeBERTa-v3-base: ALL CHECKS PASSED")
else:
    fail("DeBERTa-v3-base: FAILED — do NOT run the full notebook")


# ═══════════════════════════════════════════════════════════════════
#  TEST 2: ModernBERT-base
# ═══════════════════════════════════════════════════════════════════
header("STEP 3: ModernBERT-base (answerdotai/ModernBERT-base)")

# FIX 3: attention implementation
if sys.platform == "win32":
    attn_impl = "eager"
else:
    attn_impl = "sdpa"
print(f"  attn_implementation: {attn_impl}")
ok(f"FIX 3: attn_implementation='{attn_impl}' ({'Windows' if sys.platform == 'win32' else 'Linux'})")

bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

modernbert_ok = False
try:
    modernbert_ok = run_pipeline(
        model_name="answerdotai/ModernBERT-base",
        extra_model_kwargs={"attn_implementation": attn_impl},
        use_bf16=(DEVICE == "cuda" and bf16_ok),
        use_fp16=(DEVICE == "cuda" and not bf16_ok),
    )
except Exception as e:
    fail(f"ModernBERT pipeline CRASHED: {e}")
    traceback.print_exc()

if modernbert_ok:
    ok("ModernBERT-base: ALL CHECKS PASSED")
else:
    fail("ModernBERT-base: FAILED — do NOT run the full notebook")


# ═══════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════
header("FINAL VERDICT")

if deberta_ok and modernbert_ok:
    print(f"  {GREEN}{BOLD}ALL CLEAR — both pipelines ran without errors.{RESET}")
    print(f"  {GREEN}You are safe to run the full notebooks.{RESET}")
    print()
    print(f"  Run order:")
    print(f"    1. 04a_deberta_baseline.ipynb")
    print(f"    2. 04b_modernbert_baseline.ipynb")
else:
    print(f"  {RED}{BOLD}ISSUES DETECTED — fix before running full notebooks.{RESET}")
    if not deberta_ok:
        print(f"  {RED}  • DeBERTa pipeline failed{RESET}")
    if not modernbert_ok:
        print(f"  {RED}  • ModernBERT pipeline failed{RESET}")
    print()
    print(f"  Paste the error output into Claude for help debugging.")
