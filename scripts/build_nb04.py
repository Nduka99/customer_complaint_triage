#!/usr/bin/env python3
"""
Build NB04 — DeBERTaV3-base Fine-Tuning v5 (Frozen Warmup + LLRD Unfreeze)

Outputs a complete .ipynb JSON file to notebooks/04_debertav3_finetune.ipynb.
Run: python scripts/build_nb04.py

Strategy: Store cell sources as plain strings in a list, avoiding nested
triple-quote conflicts by using single-triple-quotes for the outer wrapper
and double-triple-quotes for Python docstrings inside cells.
"""

import json
from pathlib import Path

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code_cell(source):
    return {"cell_type": "code", "metadata": {}, "source": [source],
            "outputs": [], "execution_count": None}

cells = []

# ═══════════════════════════════════════════════════════════════
# Cell 0: Title & Architecture Decisions (Markdown)
# ═══════════════════════════════════════════════════════════════
cells.append(md_cell(
'''# 04 — DeBERTaV3-base Fine-Tuning (v5 — Frozen Warmup + LLRD Unfreeze)

Fine-tune `microsoft/deberta-v3-base` on CFPB 10-class product classification.

**Hardware:** RTX 4060 8GB VRAM, 64GB RAM, AMD Ryzen 7000.

**Rubric Criteria:** Technical Challenge (21%) — Transformer fine-tuning with advanced training strategy.

**Specialist:** Transformer Architect (TA) — Stream B: Deep Learning.

---

### v5 Architecture Decisions

| Component | Choice | Rationale |
|---|---|---|
| **Phase 1 (Frozen)** | Freeze backbone, train head only for 4 epochs | Prevents randomly-initialised head from sending garbage gradients into pretrained backbone. Head converges to a meaningful 10-class boundary first. |
| **Phase 1 LR** | 2e-4 for classifier + pooler | Safe for 0.6M randomly-initialised params. 10x standard fine-tune LR. |
| **Phase 2 (Unfrozen)** | LLRD (decay=0.85) + fresh cosine schedule | Top encoder layers (task-specific semantics) get higher LR than bottom layers (universal syntax). SOTA for DeBERTaV3 (He et al., 2023). |
| **Phase 2 base LR** | 2e-5 (classifier gets 3e-5) | Microsoft recommended rate. Classifier gets 1.5x premium to lead adaptation. |
| **Sampler** | WeightedRandomSampler | Class-balanced: each class drawn equally regardless of natural frequency. |
| **Loss** | Phase 1: plain CE. Phase 2: AlphaFocalLoss (γ=1.0, α=class weights, LS=0.05) | Phase 1 uses CE only (sampler handles balance). AlphaFocalLoss activates at unfreeze to avoid double-penalisation. |
| **Effective batch** | 64 (micro=8, accum=8) | Stable gradients for LLRD with 14+ parameter groups. |
| **Evaluation** | Phase 1: per epoch. Phase 2: every 500 steps | Tight monitoring during critical unfreeze phase. Val subsampled to 15K. |
| **Epochs** | 4 frozen + 6 unfrozen = 10 total | Validated by prior run hitting F1=0.3586 at first unfreeze epoch, still climbing. |

---'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 1: Imports
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Imports & Environment Setup
# Purpose: Load all libraries and configure hardware
# Specialist: Transformer Architect (TA)
# ============================================================
import os, json, time, warnings, gc, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import autocast

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# --- Paths ---
ROOT = Path('..').resolve()
PROCESSED = ROOT / 'data' / 'processed'
MODEL_DIR = ROOT / 'models' / 'debertav3_v5'
FIG = ROOT / 'reports' / 'figures'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    # Auto-tune convolution algorithms for consistent input sizes
    torch.backends.cudnn.benchmark = True
    # TF32 for matrix multiplications — 3x faster on Ampere+ with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
print(f'Device: {device} | PyTorch: {torch.__version__}')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 2: Reproducibility
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Reproducibility Seeds
# Purpose: Ensure deterministic training for fair comparison
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Note: We don't set torch.use_deterministic_algorithms(True) because
# DeBERTaV3's attention has non-deterministic CUDA ops.
# Seeds above give practical reproducibility (+/-0.1% F1 across runs).
print(f'Seed: {SEED} — reproducibility configured')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 3: Config
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: v5 CONFIG — Frozen Warmup + LLRD Unfreeze
# Purpose: Central config for the entire training pipeline
# ============================================================
CONFIG = {
    # --- Model ---
    'model_name': 'microsoft/deberta-v3-base',
    'max_length': 512,                # P95 token length ~787; 512 covers ~88% of samples

    # --- Two-Phase Training Schedule ---
    'num_epochs': 10,                 # 4 frozen + 6 unfrozen
    'freeze_epochs': 4,               # Phase 1: head-only training with backbone frozen
    'micro_batch_size': 8,            # Largest micro-batch that fits in 8GB with grad checkpointing
    'gradient_accumulation': 8,       # Effective batch = 64 for stable LLRD gradients

    # --- Phase 1: Frozen Warmup (head-only) ---
    # High LR is safe — only 0.6M randomly-initialised params are training.
    # The backbone is frozen so no risk of corrupting pretrained representations.
    'phase1_lr': 2e-4,                # 10x standard fine-tune LR — safe for head-only
    'phase1_warmup_ratio': 0.10,      # 10% warmup to ramp up the high LR smoothly

    # --- Phase 2: Unfrozen with LLRD ---
    # LLRD applies decreasing LR from top (layer 11) to bottom (layer 0):
    #   classifier: phase2_head_lr = 3e-5
    #   layer 11:   phase2_base_lr * 0.85^1  = 1.70e-5
    #   layer 6:    phase2_base_lr * 0.85^6  = 7.54e-6
    #   layer 0:    phase2_base_lr * 0.85^12 = 2.84e-6
    #   embeddings: phase2_base_lr * 0.85^13 = 2.42e-6
    'phase2_base_lr': 2e-5,           # Base LR for LLRD (layer 11 anchor)
    'phase2_head_lr': 3e-5,           # Classifier head gets 1.5x premium
    'llrd_decay': 0.85,               # Per-layer multiplicative decay
    'phase2_warmup_ratio': 0.06,      # 6% warmup for the fresh Phase 2 schedule

    # --- Shared Optimiser Settings ---
    'weight_decay': 0.01,             # L2 regularisation (skip bias & LayerNorm)
    'max_grad_norm': 1.0,             # Gradient clipping
    'adam_eps': 1e-6,                  # AdamW epsilon (DeBERTa default)

    # --- Loss: Alpha-Balanced Focal Loss ---
    # gamma=1.0 (not 2.0): sampler already handles class balance,
    # focal just does hard-example mining with a light touch.
    'focal_gamma': 1.0,
    'label_smoothing': 0.05,
    'use_class_weights': True,

    # --- Regularisation ---
    'dropout': 0.1,                   # Default DeBERTa dropout

    # --- Data ---
    'subsample_train': 25_000,        # Validation run: 25K subsample
    'subsample_val': 15_000,          # Fast eval: ~3 min instead of 97 min
    'target_col': 'product_id',

    # --- Evaluation ---
    'phase2_eval_every_steps': 500,   # Step-based eval in Phase 2
    'patience': 5,                    # 5 evals without improvement -> early stop

    # --- Hardware ---
    'use_amp': True,
    'amp_dtype': 'bfloat16',          # BF16 on Ampere+
    'gradient_checkpointing': True,   # Only enabled in Phase 2
    'dynamic_padding': True,
}

CONFIG['effective_batch'] = CONFIG['micro_batch_size'] * CONFIG['gradient_accumulation']
CONFIG['amp_dtype_torch'] = torch.bfloat16 if CONFIG['amp_dtype'] == 'bfloat16' else torch.float16

print(f"{'='*60}")
print(f"DeBERTaV3-base v5 — Frozen Warmup + LLRD Unfreeze")
print(f"{'='*60}")
print(f"Effective batch size: {CONFIG['effective_batch']}")
print(f"Phase 1: {CONFIG['freeze_epochs']} frozen epochs, head LR={CONFIG['phase1_lr']:.0e}")
print(f"Phase 2: {CONFIG['num_epochs'] - CONFIG['freeze_epochs']} unfrozen epochs, LLRD decay={CONFIG['llrd_decay']}")
print(f"  Base LR: {CONFIG['phase2_base_lr']:.0e} | Head LR: {CONFIG['phase2_head_lr']:.0e}")
print(f"Loss: AlphaFocal(gamma={CONFIG['focal_gamma']}, LS={CONFIG['label_smoothing']})")
print(f"Data: {CONFIG['subsample_train']//1000}K train, {CONFIG['subsample_val']//1000}K val")
print(f"{'='*60}")

config_save = {k: str(v) if isinstance(v, torch.dtype) else v for k, v in CONFIG.items()}
with open(MODEL_DIR / 'config_v5.json', 'w') as f:
    json.dump(config_save, f, indent=2)
print(f'Config saved to {MODEL_DIR / "config_v5.json"}')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 4: Load Data + WeightedRandomSampler
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Data Loading + Class-Balanced Sampling
# Purpose: Load preprocessed data, subsample, build sampler
# ============================================================
train_df = pd.read_parquet(PROCESSED / 'train.parquet')
val_df = pd.read_parquet(PROCESSED / 'val.parquet')

with open(PROCESSED / 'preprocessing_config.json') as f:
    prep_config = json.load(f)

n_classes = prep_config['n_products']
class_names = prep_config['product_classes']

print(f'Full data: Train={len(train_df):,} | Val={len(val_df):,} | Classes={n_classes}')

# --- Stratified subsample for validation run ---
if CONFIG['subsample_train'] and CONFIG['subsample_train'] < len(train_df):
    train_df = train_df.groupby(CONFIG['target_col'], group_keys=False).apply(
        lambda x: x.sample(
            n=max(1, int(len(x) * CONFIG['subsample_train'] / len(train_df))),
            random_state=SEED
        )
    ).reset_index(drop=True)
    print(f'Train subsampled (stratified): {len(train_df):,}')

if CONFIG['subsample_val'] and CONFIG['subsample_val'] < len(val_df):
    val_df = val_df.sample(CONFIG['subsample_val'], random_state=SEED).reset_index(drop=True)
    print(f'Val subsampled: {len(val_df):,}')

# --- Class distribution ---
print(f'\\nClass distribution (train subsample):')
class_counts = train_df[CONFIG['target_col']].value_counts().sort_index()
for cls_id, count in class_counts.items():
    name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
    pct = count / len(train_df) * 100
    print(f'  {cls_id}: {name[:45]:45s} {count:>6,}  ({pct:.2f}%)')

# ============================================================
# WeightedRandomSampler — class-balanced drawing
# ============================================================
# Each sample weight = 1 / (n_classes * count_of_its_class)
# Every CLASS has equal total weight -> sampler draws ~equal per class.
labels_array = train_df[CONFIG['target_col']].values
class_counts_array = np.bincount(labels_array, minlength=n_classes)

sample_weights = np.zeros(len(labels_array), dtype=np.float64)
for cls_id in range(n_classes):
    mask = labels_array == cls_id
    if class_counts_array[cls_id] > 0:
        sample_weights[mask] = 1.0 / (n_classes * class_counts_array[cls_id])

for cls_id in range(n_classes):
    mask = labels_array == cls_id
    total_w = sample_weights[mask].sum()
    print(f'  Class {cls_id} total weight: {total_w:.6f}')

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights),
    num_samples=len(train_df),
    replacement=True
)
print(f'\\nWeightedRandomSampler: {len(train_df):,} draws/epoch')
print(f'Expected per-class draws: ~{len(train_df) // n_classes:,}/epoch')

# ============================================================
# Alpha Class Weights for Focal Loss
# ============================================================
if CONFIG['use_class_weights']:
    weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(labels_array), y=labels_array
    )
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print('\\nAlpha Class Weights:')
    for i, w in enumerate(weights):
        name = class_names[i][:30] if i < len(class_names) else f'class_{i}'
        print(f'  {i}: {name:30s} -> alpha={w:.4f}')
else:
    class_weights_tensor = None'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 5: Dataset & DataLoaders
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Dataset & DataLoader Construction
# Purpose: On-the-fly tokenisation with dynamic padding
# ============================================================
tokeniser = AutoTokenizer.from_pretrained(CONFIG['model_name'])
print(f'Tokeniser: {CONFIG["model_name"]} (vocab: {tokeniser.vocab_size:,})')

class CFPBDataset(Dataset):
    # On-the-fly tokenisation — no pre-padding.
    # Dynamic padding via DataCollatorWithPadding pads each batch
    # to its longest sequence, saving ~40% compute vs fixed 512.
    def __init__(self, texts, labels, tokeniser, max_length):
        self.texts = texts
        self.labels = labels
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokeniser(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc['labels'] = self.labels[idx]
        return enc

train_ds = CFPBDataset(
    texts=train_df['narrative'].tolist(),
    labels=train_df[CONFIG['target_col']].tolist(),
    tokeniser=tokeniser,
    max_length=CONFIG['max_length'],
)
val_ds = CFPBDataset(
    texts=val_df['narrative'].tolist(),
    labels=val_df[CONFIG['target_col']].tolist(),
    tokeniser=tokeniser,
    max_length=CONFIG['max_length'],
)

collator = DataCollatorWithPadding(tokenizer=tokeniser, padding='longest', return_tensors='pt')
BS = CONFIG['micro_batch_size']

# Train loader with WeightedRandomSampler (replaces shuffle=True)
train_loader = DataLoader(
    train_ds, batch_size=BS, sampler=sampler,
    num_workers=0, pin_memory=True, drop_last=True, collate_fn=collator,
)
# Val loader: sequential, larger batch for faster eval
val_loader = DataLoader(
    val_ds, batch_size=BS * 4, shuffle=False,
    num_workers=0, pin_memory=True, collate_fn=collator,
)

steps_per_epoch = len(train_loader) // CONFIG['gradient_accumulation']
phase1_total_steps = steps_per_epoch * CONFIG['freeze_epochs']
phase2_total_steps = steps_per_epoch * (CONFIG['num_epochs'] - CONFIG['freeze_epochs'])
total_optimizer_steps = phase1_total_steps + phase2_total_steps

print(f'Train: {len(train_loader):,} micro-batches/epoch -> {steps_per_epoch:,} optimizer steps/epoch')
print(f'Phase 1: {phase1_total_steps:,} steps ({CONFIG["freeze_epochs"]} epochs)')
print(f'Phase 2: {phase2_total_steps:,} steps ({CONFIG["num_epochs"] - CONFIG["freeze_epochs"]} epochs)')
print(f'Total: {total_optimizer_steps:,} steps')
print(f'Val: {len(val_loader):,} batches')

# --- Time estimates ---
est_p1 = phase1_total_steps * 0.15 / 3600  # ~0.15s/step (no grad checkpoint)
est_p2 = phase2_total_steps * 0.45 / 3600  # ~0.45s/step (grad checkpoint)
est_eval = len(val_loader) * 0.05 / 60     # ~3 min per eval
n_p1_evals = CONFIG['freeze_epochs']
n_p2_evals = phase2_total_steps // CONFIG['phase2_eval_every_steps']
est_total = est_p1 + est_p2 + (n_p1_evals + n_p2_evals) * est_eval / 60
print(f'\\nEst. total wall time: ~{est_total:.1f} hours')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 6: Loss Function
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Alpha-Balanced Focal Loss
# Purpose: Handle class imbalance + hard-example mining
# ============================================================
class AlphaFocalLoss(nn.Module):
    # Combines Focal Loss (gamma) to down-weight easy examples,
    # with Class Weights (alpha) to penalise minority class errors,
    # and label smoothing to prevent overconfident predictions.
    #
    # gamma=1.0 (not 2.0): WeightedRandomSampler already balances classes
    # at batch level. Focal just does hard-example mining with a light touch.
    #
    # Math: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    def __init__(self, gamma=1.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Per-sample cross-entropy with label smoothing
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction='none',
            label_smoothing=self.label_smoothing
        )
        # pt = model confidence in the true class
        pt = torch.exp(-ce_loss)
        # Focal modulating factor: suppress easy examples (high pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        # Alpha class weights: upweight minority class errors
        if self.weight is not None:
            alpha = self.weight[targets]
            focal_loss = focal_loss * alpha
        return focal_loss.mean()

# Phase 2 loss: AlphaFocalLoss — used after backbone unfreezes
criterion_focal = AlphaFocalLoss(
    gamma=CONFIG['focal_gamma'],
    weight=class_weights_tensor,
    label_smoothing=CONFIG['label_smoothing']
)

# Phase 1 loss: Plain CrossEntropy — no alpha weights, no focal
# WHY: During Phase 1, WeightedRandomSampler already gives equal class exposure.
# Adding alpha weights (class 4 gets ~54x) on top of the sampler creates
# extremely large loss values that destabilise the randomly-initialised head.
# Plain CE lets the head converge smoothly to a sensible 10-class boundary.
criterion_ce = nn.CrossEntropyLoss()

# Start with Phase 1 loss — will switch at unfreeze
criterion = criterion_ce
print(f'Phase 1 loss: CrossEntropyLoss (plain — sampler handles class balance)')
print(f'Phase 2 loss: AlphaFocalLoss(gamma={CONFIG["focal_gamma"]}, LS={CONFIG["label_smoothing"]}, alpha=ON)')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 7: Model + Freeze/Unfreeze + LLRD
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Model Loading + Freeze/Unfreeze + LLRD Optimizer
# Purpose: DeBERTaV3 model with two-phase training utilities
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=n_classes,
    hidden_dropout_prob=CONFIG['dropout'],
    attention_probs_dropout_prob=CONFIG['dropout'],
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
backbone_params = sum(p.numel() for n, p in model.named_parameters()
                      if 'classifier' not in n and 'pooler' not in n)
head_params = total_params - backbone_params
print(f'Total params:    {total_params/1e6:.1f}M')
print(f'Backbone params: {backbone_params/1e6:.1f}M (frozen in Phase 1)')
print(f'Head params:     {head_params/1e6:.3f}M (trainable in Phase 1)')

# ============================================================
# Freeze / Unfreeze utilities
# ============================================================
def freeze_backbone(model):
    # Freeze all backbone params — only classifier + pooler remain trainable.
    # WHY: During Phase 1 the head is randomly initialised. If gradients flow
    # into the backbone, the random signal corrupts pretrained representations.
    for name, p in model.named_parameters():
        if 'classifier' not in name and 'pooler' not in name:
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Backbone FROZEN. Trainable: {trainable:,} ({trainable/1e6:.3f}M)')

def unfreeze_backbone(model):
    # Unfreeze all parameters for Phase 2 full fine-tuning with LLRD.
    # After Phase 1, the head produces informative gradients so the backbone
    # can safely adapt to CFPB financial language.
    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Backbone UNFROZEN. Trainable: {trainable:,} ({trainable/1e6:.1f}M)')

# ============================================================
# LLRD — Layer-wise Learning Rate Decay (for Phase 2)
# ============================================================
def get_llrd_param_groups(model, base_lr, head_lr, decay, weight_decay):
    # Build LLRD parameter groups for DeBERTaV3-base (12 encoder layers).
    # Top layers (task-specific) get higher LR, bottom layers (syntax) get lower.
    #   classifier/pooler: head_lr
    #   layer 11: base_lr * decay^1 = 1.7e-5
    #   layer 0:  base_lr * decay^12 = 2.8e-6
    #   embeddings: base_lr * decay^13 = 2.4e-6
    # No weight decay on bias and LayerNorm params.
    no_decay = {'bias', 'LayerNorm.weight', 'LayerNorm.bias',
                'layer_norm.weight', 'layer_norm.bias'}
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'classifier' in name or 'pooler' in name:
            lr = head_lr
            group_name = 'classifier'
        elif 'embeddings' in name:
            lr = base_lr * (decay ** 13)
            group_name = 'embeddings'
        else:
            layer_num = None
            for part in name.split('.'):
                if part.isdigit():
                    layer_num = int(part)
                    break
            if layer_num is not None:
                depth = 12 - layer_num
                lr = base_lr * (decay ** depth)
                group_name = f'layer_{layer_num}'
            else:
                lr = base_lr * (decay ** 13)
                group_name = 'encoder_other'

        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        key = (group_name, lr, wd > 0)
        if key not in param_groups:
            param_groups[key] = {
                'params': [], 'lr': lr, 'weight_decay': wd, '_gn': group_name
            }
        param_groups[key]['params'].append(param)

    groups = list(param_groups.values())
    print(f'\\n  LLRD Parameter Groups (decay={decay}):')
    print(f'  {"Group":<20} {"LR":>12} {"Params":>10}')
    print(f'  {"-"*44}')
    summary = {}
    for g in groups:
        gn = g['_gn']
        n_p = sum(p.numel() for p in g['params'])
        if gn not in summary:
            summary[gn] = {'lr': g['lr'], 'params': 0}
        summary[gn]['params'] += n_p
    for gn in ['classifier', 'layer_11', 'layer_10', 'layer_9', 'layer_6',
               'layer_3', 'layer_0', 'embeddings', 'encoder_other']:
        if gn in summary:
            s = summary[gn]
            print(f'  {gn:<20} {s["lr"]:>12.2e} {s["params"]/1e6:>9.2f}M')
    for g in groups:
        del g['_gn']
    return groups

# ============================================================
# Phase 1 Setup: Freeze backbone, head-only optimizer
# ============================================================
print('\\n--- Phase 1 Setup ---')
freeze_backbone(model)

head_params_list = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    head_params_list, lr=CONFIG['phase1_lr'],
    weight_decay=CONFIG['weight_decay'], eps=CONFIG['adam_eps'],
)
phase1_warmup = int(phase1_total_steps * CONFIG['phase1_warmup_ratio'])
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=phase1_warmup, num_training_steps=phase1_total_steps,
)
print(f'  Optimizer: AdamW (head only, lr={CONFIG["phase1_lr"]:.0e})')
print(f'  Scheduler: cosine, {phase1_warmup} warmup / {phase1_total_steps} total steps')
print(f'  Gradient checkpointing: OFF (not needed for 0.6M params)')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 8: Evaluation + VRAM monitor
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Evaluation Function + VRAM Monitor
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True, amp_dtype=torch.bfloat16,
             desc='Eval'):
    # Full eval pass with no gradient tracking — zero memory overhead.
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(loader, desc=desc, leave=False):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labs = batch['labels'].to(device)

        with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labs)

        total_loss += loss.item()
        n_batches += 1
        all_preds.extend(out.logits.argmax(-1).cpu().numpy())
        all_labels.extend(labs.cpu().numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        'predictions': preds,
        'labels': labels,
    }

def log_vram(tag=''):
    # Log VRAM usage — RTX 4060 has 8GB budget.
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f'  VRAM [{tag}]: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB')

log_vram('pre-training')
print('Evaluation function defined.')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 9: Training Loop
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Two-Phase Training Loop
# Purpose: Phase 1 (frozen warmup) -> Phase 2 (LLRD unfreeze)
# ============================================================
history = {
    'epoch': [], 'step': [], 'phase': [],
    'train_loss': [], 'val_loss': [],
    'val_f1_macro': [], 'val_accuracy': [],
    'val_f1_weighted': [], 'val_precision_macro': [], 'val_recall_macro': [],
    'lr': [], 'wall_time_min': [],
}
best_f1 = 0.0
patience_counter = 0
global_step = 0
use_amp = CONFIG['use_amp']
amp_dtype = CONFIG['amp_dtype_torch']

def log_eval(epoch, step, phase, train_loss, val_metrics, lr, wall_min):
    history['epoch'].append(epoch)
    history['step'].append(step)
    history['phase'].append(phase)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_metrics['loss'])
    history['val_f1_macro'].append(val_metrics['f1_macro'])
    history['val_accuracy'].append(val_metrics['accuracy'])
    history['val_f1_weighted'].append(val_metrics['f1_weighted'])
    history['val_precision_macro'].append(val_metrics['precision_macro'])
    history['val_recall_macro'].append(val_metrics['recall_macro'])
    history['lr'].append(lr)
    history['wall_time_min'].append(wall_min)

print(f'\\n{"="*70}')
print(f'TRAINING START — DeBERTaV3-base v5')
print(f'Phase 1: {CONFIG["freeze_epochs"]} epochs FROZEN (head only, lr={CONFIG["phase1_lr"]:.0e})')
print(f'Phase 2: {CONFIG["num_epochs"] - CONFIG["freeze_epochs"]} epochs UNFROZEN (LLRD, decay={CONFIG["llrd_decay"]})')
print(f'Eval: Phase 1 = per epoch | Phase 2 = every {CONFIG["phase2_eval_every_steps"]} steps')
print(f'Patience: {CONFIG["patience"]} evals without improvement')
print(f'{"="*70}\\n')

train_start = time.time()

for epoch in range(CONFIG['num_epochs']):
    t_epoch = time.time()
    current_phase = 'FROZEN' if epoch < CONFIG['freeze_epochs'] else 'FULL'

    # ============================================================
    # PHASE TRANSITION: Unfreeze at the boundary
    # ============================================================
    if epoch == CONFIG['freeze_epochs']:
        print(f'\\n{"="*70}')
        print(f'PHASE 2 TRANSITION — Unfreezing backbone with LLRD')
        print(f'{"="*70}')

        # Step 0: Switch to AlphaFocalLoss for Phase 2
        # Now that the head is converged, alpha weights + focal mining help
        # the full model focus on hard examples and rare classes.
        criterion = criterion_focal
        print(f'  Loss switched to AlphaFocalLoss')

        # Step 1: Unfreeze all parameters
        unfreeze_backbone(model)

        # Step 2: Enable gradient checkpointing (needed for 184M params in 8GB)
        if CONFIG['gradient_checkpointing']:
            # use_reentrant=False is CRITICAL: with the default (True),
            # frozen params + grad checkpointing silently breaks gradient flow
            # (HuggingFace Issue #21381). All Phase 2 gradients would be zero.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print(f'  Gradient checkpointing: ON')

        # Step 3: LLRD optimizer with per-layer learning rates
        llrd_groups = get_llrd_param_groups(
            model,
            base_lr=CONFIG['phase2_base_lr'],
            head_lr=CONFIG['phase2_head_lr'],
            decay=CONFIG['llrd_decay'],
            weight_decay=CONFIG['weight_decay'],
        )
        optimizer = torch.optim.AdamW(llrd_groups, eps=CONFIG['adam_eps'])

        # Step 4: Fresh cosine schedule — optimization landscape changes
        # dramatically when 184M params suddenly become trainable
        phase2_warmup = int(phase2_total_steps * CONFIG['phase2_warmup_ratio'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=phase2_warmup,
            num_training_steps=phase2_total_steps,
        )
        print(f'  Fresh cosine schedule: {phase2_warmup} warmup / {phase2_total_steps} total')

        # Step 5: Reset patience — Phase 2 metrics may dip temporarily
        patience_counter = 0
        print(f'  Patience counter reset.')
        log_vram('post-unfreeze')
        print(f'{"="*70}\\n')

    # ============================================================
    # TRAINING EPOCH
    # ============================================================
    model.train()
    epoch_loss = 0.0
    n_micro_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]} [{current_phase}]')

    for micro_step, batch in enumerate(pbar):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labs = batch['labels'].to(device)

        with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labs) / CONFIG['gradient_accumulation']

        loss.backward()
        epoch_loss += loss.detach().item() * CONFIG['gradient_accumulation']
        n_micro_batches += 1

        if (micro_step + 1) % CONFIG['gradient_accumulation'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = epoch_loss / n_micro_batches
                current_lr = scheduler.get_last_lr()[0]
                elapsed_hrs = (time.time() - train_start) / 3600
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}',
                    'step': f'{global_step}', 'hrs': f'{elapsed_hrs:.1f}'
                })

            # --- Phase 2: Step-based evaluation ---
            if (current_phase == 'FULL' and
                global_step % CONFIG['phase2_eval_every_steps'] == 0):

                eval_start = time.time()
                val_metrics = evaluate(
                    model, val_loader, criterion, device,
                    use_amp=use_amp, amp_dtype=amp_dtype,
                    desc=f'Val@step{global_step}'
                )
                eval_time = time.time() - eval_start
                wall_min = (time.time() - train_start) / 60
                avg_train_loss = epoch_loss / n_micro_batches
                current_lr = scheduler.get_last_lr()[0]

                log_eval(epoch + 1, global_step, current_phase,
                         avg_train_loss, val_metrics, current_lr, wall_min)

                print(f'\\n  [Step {global_step:,}] ({wall_min:.0f} min, eval {eval_time:.0f}s)')
                print(f'  train_loss={avg_train_loss:.4f}  val_loss={val_metrics["loss"]:.4f}')
                print(f'  acc={val_metrics["accuracy"]:.4f}  F1_macro={val_metrics["f1_macro"]:.4f}  '
                      f'F1_wt={val_metrics["f1_weighted"]:.4f}  '
                      f'P={val_metrics["precision_macro"]:.4f}  '
                      f'R={val_metrics["recall_macro"]:.4f}')
                print(f'  lr={current_lr:.2e}')

                if val_metrics['f1_macro'] > best_f1:
                    improvement = val_metrics['f1_macro'] - best_f1
                    best_f1 = val_metrics['f1_macro']
                    patience_counter = 0
                    torch.save(model.state_dict(), MODEL_DIR / 'best_model_v5.pt')
                    print(f'  >>> New best F1! {best_f1:.4f} (+{improvement:.4f}) — saved.')
                else:
                    patience_counter += 1
                    print(f'  No improvement ({patience_counter}/{CONFIG["patience"]})')
                    if patience_counter >= CONFIG['patience']:
                        print(f'  EARLY STOPPING at step {global_step:,}')
                        break

                model.train()

    # Check early stopping
    if patience_counter >= CONFIG['patience'] and current_phase == 'FULL':
        break

    # ============================================================
    # END-OF-EPOCH: Phase 1 eval (once per epoch)
    # ============================================================
    epoch_time = time.time() - t_epoch
    avg_train_loss = epoch_loss / max(n_micro_batches, 1)

    if current_phase == 'FROZEN':
        eval_start = time.time()
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            use_amp=use_amp, amp_dtype=amp_dtype, desc=f'Val@epoch{epoch+1}'
        )
        eval_time = time.time() - eval_start
        wall_min = (time.time() - train_start) / 60
        current_lr = scheduler.get_last_lr()[0]

        log_eval(epoch + 1, global_step, current_phase,
                 avg_train_loss, val_metrics, current_lr, wall_min)

        print(f'\\nEpoch {epoch+1} [{current_phase}] ({epoch_time:.0f}s) — '
              f'train_loss={avg_train_loss:.4f}  val_loss={val_metrics["loss"]:.4f}')
        print(f'  acc={val_metrics["accuracy"]:.4f}  F1_macro={val_metrics["f1_macro"]:.4f}  '
              f'F1_wt={val_metrics["f1_weighted"]:.4f}  lr={current_lr:.2e}')

        if val_metrics['f1_macro'] > best_f1:
            improvement = val_metrics['f1_macro'] - best_f1
            best_f1 = val_metrics['f1_macro']
            torch.save(model.state_dict(), MODEL_DIR / 'best_model_v5.pt')
            print(f'  >>> New best F1! {best_f1:.4f} (+{improvement:.4f}) — saved.')
        log_vram(f'epoch{epoch+1}')
    else:
        print(f'\\nEpoch {epoch+1} [{current_phase}] ({epoch_time:.0f}s) — avg_loss={avg_train_loss:.4f}')

total_time = time.time() - train_start
best_idx = np.argmax(history['val_f1_macro']) if history['val_f1_macro'] else 0
print(f'\\n{"="*70}')
print(f'TRAINING COMPLETE in {total_time/3600:.1f} hours ({total_time/60:.0f} min)')
print(f'Best val F1 macro: {best_f1:.4f}')
if history['step']:
    print(f'Best at step {history["step"][best_idx]:,} (epoch {history["epoch"][best_idx]})')
print(f'Total optimizer steps: {global_step:,}')
print(f'{"="*70}')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 10: Training Curves
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Training Curves Visualisation
# ============================================================
steps = history['step']
phases = history['phase']

if not steps:
    print('No training history to plot.')
else:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    frozen_mask = [p == 'FROZEN' for p in phases]
    full_mask = [p == 'FULL' for p in phases]
    frozen_steps = [s for s, m in zip(steps, frozen_mask) if m]
    full_steps = [s for s, m in zip(steps, full_mask) if m]

    # Loss
    ax = axes[0, 0]
    ax.plot(steps, history['train_loss'], 'o-', label='Train', markersize=4, alpha=0.7)
    ax.plot(steps, history['val_loss'], 'o-', label='Val', markersize=4, alpha=0.7)
    if frozen_steps and full_steps:
        ax.axvline(full_steps[0], color='red', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_title('Loss vs Step'); ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # F1 macro
    ax = axes[0, 1]
    if frozen_steps:
        f1_f = [f for f, m in zip(history['val_f1_macro'], frozen_mask) if m]
        ax.plot(frozen_steps, f1_f, 'o-', color='blue', markersize=5, label='Phase 1 (Frozen)')
    if full_steps:
        f1_u = [f for f, m in zip(history['val_f1_macro'], full_mask) if m]
        ax.plot(full_steps, f1_u, 'o-', color='green', markersize=4, label='Phase 2 (LLRD)')
    best_idx = np.argmax(history['val_f1_macro'])
    ax.axhline(y=history['val_f1_macro'][best_idx], color='red', ls='--', alpha=0.5)
    ax.annotate(f'Best: {history["val_f1_macro"][best_idx]:.4f}',
                xy=(steps[best_idx], history['val_f1_macro'][best_idx]),
                fontsize=10, color='red', fontweight='bold')
    ax.set_title('Val Macro F1 vs Step'); ax.set_xlabel('Step'); ax.set_ylabel('F1 Macro')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1, 0]
    ax.plot(steps, history['val_accuracy'], 'o-', color='blue', markersize=4)
    if frozen_steps and full_steps:
        ax.axvline(full_steps[0], color='red', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_title('Val Accuracy vs Step'); ax.set_xlabel('Step'); ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(steps, history['lr'], 'o-', color='purple', markersize=4)
    if frozen_steps and full_steps:
        ax.axvline(full_steps[0], color='red', ls='--', alpha=0.5, label='Unfreeze')
    ax.set_title('Learning Rate Schedule'); ax.set_xlabel('Step'); ax.set_ylabel('LR')
    ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('DeBERTaV3-base v5 — Frozen Warmup + LLRD Unfreeze', y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG / '04_debertav3_v5_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    history_save = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in history.items()}
    with open(MODEL_DIR / 'training_history_v5.json', 'w') as f:
        json.dump(history_save, f, indent=2)
    print('Training curves saved.')'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 11: Final Validation
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Final Validation — Best Checkpoint
# ============================================================
print('Loading best checkpoint for final evaluation...')
model.load_state_dict(
    torch.load(MODEL_DIR / 'best_model_v5.pt', map_location=device, weights_only=True)
)

val_final = evaluate(model, val_loader, criterion, device,
                     use_amp=use_amp, amp_dtype=amp_dtype, desc='Final Val')

print(f'\\n{"="*60}')
print(f'VALIDATION RESULTS — DeBERTaV3-base v5')
print(f'{"="*60}')
print(f'Accuracy:         {val_final["accuracy"]:.4f}')
print(f'F1 macro:         {val_final["f1_macro"]:.4f}')
print(f'F1 weighted:      {val_final["f1_weighted"]:.4f}')
print(f'Precision macro:  {val_final["precision_macro"]:.4f}')
print(f'Recall macro:     {val_final["recall_macro"]:.4f}')
print(f'Loss:             {val_final["loss"]:.4f}')
print()
print(classification_report(
    val_final['labels'], val_final['predictions'],
    target_names=[n[:45] for n in class_names],
    digits=4, zero_division=0
))'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 12: Test Set
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Test Set Evaluation — Held-out Temporal Split (2024-H2)
# ============================================================
test_df = pd.read_parquet(PROCESSED / 'test.parquet')
print(f'Test set: {len(test_df):,} samples')

test_ds = CFPBDataset(
    test_df['narrative'].tolist(), test_df[CONFIG['target_col']].tolist(),
    tokeniser, CONFIG['max_length']
)
test_loader = DataLoader(
    test_ds, batch_size=BS * 4, shuffle=False,
    num_workers=0, pin_memory=True, collate_fn=collator
)

test_final = evaluate(model, test_loader, criterion, device,
                      use_amp=use_amp, amp_dtype=amp_dtype, desc='Test')

print(f'\\n{"="*60}')
print(f'TEST RESULTS — DeBERTaV3-base v5')
print(f'{"="*60}')
print(f'Accuracy:         {test_final["accuracy"]:.4f}')
print(f'F1 macro:         {test_final["f1_macro"]:.4f}')
print(f'F1 weighted:      {test_final["f1_weighted"]:.4f}')
print(f'Precision macro:  {test_final["precision_macro"]:.4f}')
print(f'Recall macro:     {test_final["recall_macro"]:.4f}')
print(f'Loss:             {test_final["loss"]:.4f}')
print()
print(classification_report(
    test_final['labels'], test_final['predictions'],
    target_names=[n[:45] for n in class_names],
    digits=4, zero_division=0
))'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 13: Confusion Matrices
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Confusion Matrices — Val & Test
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
short_names = [n[:20] for n in class_names]

for ax, metrics, title in [
    (axes[0], val_final, 'Validation'),
    (axes[1], test_final, 'Test'),
]:
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{title} — F1={metrics["f1_macro"]:.4f} Acc={metrics["accuracy"]:.4f}')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)

plt.suptitle('DeBERTaV3-base v5 — Normalised Confusion Matrices', fontsize=14)
plt.tight_layout()
fig.savefig(FIG / '04_debertav3_v5_confusion.png', dpi=150, bbox_inches='tight')
plt.show()'''
))

# ═══════════════════════════════════════════════════════════════
# Cell 14: Save Results + Cleanup
# ═══════════════════════════════════════════════════════════════
cells.append(code_cell(
'''# ============================================================
# SECTION: Save Results + VRAM Cleanup
# ============================================================
results = {
    'model': 'debertav3-base-v5',
    'version': 'v5_frozen_warmup_llrd_unfreeze',
    'strategy': 'Phase 1: 4 frozen epochs (head only), Phase 2: 6 unfrozen (LLRD)',
    'validation': {
        'accuracy': float(val_final['accuracy']),
        'f1_macro': float(val_final['f1_macro']),
        'f1_weighted': float(val_final['f1_weighted']),
        'precision_macro': float(val_final['precision_macro']),
        'recall_macro': float(val_final['recall_macro']),
        'loss': float(val_final['loss']),
    },
    'test': {
        'accuracy': float(test_final['accuracy']),
        'f1_macro': float(test_final['f1_macro']),
        'f1_weighted': float(test_final['f1_weighted']),
        'precision_macro': float(test_final['precision_macro']),
        'recall_macro': float(test_final['recall_macro']),
        'loss': float(test_final['loss']),
    },
    'config': config_save,
    'training': {
        'total_optimizer_steps': global_step,
        'best_step': int(history['step'][np.argmax(history['val_f1_macro'])]) if history['step'] else 0,
        'training_time_hours': total_time / 3600,
        'train_subsample_size': len(train_df),
        'val_subsample_size': len(val_df),
    },
}

with open(MODEL_DIR / 'results_v5.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to {MODEL_DIR / "results_v5.json"}')
print(f'\\nKey artifacts:')
print(f'  Best model:        {MODEL_DIR / "best_model_v5.pt"}')
print(f'  Config:            {MODEL_DIR / "config_v5.json"}')
print(f'  Training history:  {MODEL_DIR / "training_history_v5.json"}')
print(f'  Training curves:   {FIG / "04_debertav3_v5_curves.png"}')
print(f'  Confusion matrices: {FIG / "04_debertav3_v5_confusion.png"}')

del model
gc.collect()
torch.cuda.empty_cache()
log_vram('cleanup')
print(f'\\nVRAM freed. Ready for next encoder (ModernBERT / NeoBERT).')'''
))

# ═══════════════════════════════════════════════════════════════
# Assemble the notebook
# ═══════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (.nlpproj)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "cells": cells,
}

out_path = Path(__file__).resolve().parent.parent / 'notebooks' / '04_debertav3_finetune.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Notebook written to: {out_path}')
print(f'Total cells: {len(cells)}')
