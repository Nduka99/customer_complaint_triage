/**
 * Pre-cached example results — shown while the backend is waking from cold start.
 *
 * These are real responses from the pipeline, captured locally.
 * They let an employer see what the system does immediately, even if HF Spaces
 * is still booting. Displayed with a "Cached" badge so it's honest.
 *
 * CACHED_EXAMPLE: High-confidence automated routing (green badge)
 * CACHED_ESCALATION: Low-confidence forced escalation (amber badge)
 *
 * Update these whenever the pipeline output shape changes.
 */

export const CACHED_EXAMPLE = {
  summary: "Classified as: Checking or savings account (92.4%)",
  classification: {
    label: "Checking or savings account",
    confidence: 0.9243606692139804,
  },
  routing: {
    decision: "Ensemble (Stacked)",
    arm: 2,
    reason: "Thompson samples: [0.803, 0.558, 0.878, 0.545]",
  },
  rag_context: [
    {
      text: "required for EFTs of $15 or less (12 CFR 1005.9(e)). Periodic Statements. Periodic statements must be sent for each monthly cycle in which an EFT has occurred, and at least quarterly if no EFT has occurred (12 CFR 1005.9(b)). For each EFT made during the cycle, the statement must include, as applicable: amount of the transfer, date the transfer was posted to the account.",
      source: "CFPB EFTA Examination Procedures (Deposit Accounts)",
      issue: "Problem with a lender or other company charging your account",
    },
    {
      text: "in which pre-payment disclosures and receipts are provided that do not contain estimates, confirm with respect to any transaction for which payment was made, that the information on the most recent pre-payment disclosure for that transaction and the information on the receipt for that transaction are the same.",
      source: "CFPB EFTA Examination Procedures (Deposit Accounts)",
      issue: "Problem with a lender or other company charging your account",
    },
    {
      text: "an account is scheduled to be credited by a preauthorized EFT from the same payor at least once every 60 days, the financial institution must provide some form of notice to the consumer so that the consumer can find out whether or not the transfer occurred (12 CFR 1005.10(a)).",
      source: "CFPB EFTA Examination Procedures (Deposit Accounts)",
      issue: "Problem with a credit reporting company's investigation into an existing problem",
    },
  ],
  agentic_trace: [
    "1. Models loaded from HF Hub (cached).",
    "2. RoBERTa-D inference: Checking or savings account (0.99)",
    "3. ModernBERT inference: Checking or savings account (1.00)",
    "4. LR Stacker decision: Checking or savings account (0.92)",
    "5. Thompson Sampling: Arm 2 (Ensemble (Stacked)) selected",
    "6. RAG retrieved 3 passages for 'Checking or savings account'",
  ],
};

// Escalation example — models disagree, confidence drops below 0.55, forced to human review.
// RoBERTa says "Debt/credit management", ModernBERT says "Debt collection" — the #1 confused
// pair from NB09 (28.7% misroute rate). Stacker confidence = 0.398, well below 0.55 threshold.
export const CACHED_ESCALATION = {
  summary: "Classified as: Debt or credit management (39.8%)",
  classification: {
    label: "Debt or credit management",
    confidence: 0.3983,
  },
  routing: {
    decision: "Human Escalation",
    arm: 3,
    reason: "Confidence 0.398 below 0.55 threshold — forced escalation",
  },
  rag_context: [
    {
      text: "constraints, use service providers to develop and market additional products or services, or rely on expertise from service providers that would not otherwise be available without significant investment. Service provider relationships may pose risks, however, and the CFPB expects supervised institutions to have an effective process for managing those risks.",
      source: "CFPB Compliance Management Review Examination Procedures",
      issue: "Problem with a company's investigation into an existing problem",
    },
    {
      text: "firm, legal entity, division, or business unit in the way that is most effective for the institution, and that the manner of organization will vary from institution to institution. The compliance management system should be commensurate with the entity's size, complexity, and risk profile.",
      source: "CFPB Compliance Management Review Examination Procedures",
      issue: "Problem with a lender or other company charging your account",
    },
    {
      text: "The board of directors or an appropriate board committee is expected to provide oversight of the compliance management system, including compliance audit. The board should be knowledgeable about consumer compliance requirements applicable to the institution's activities.",
      source: "CFPB Compliance Management Review Examination Procedures",
      issue: "Credit monitoring or identity theft protection services",
    },
  ],
  agentic_trace: [
    "1. Models loaded from HF Hub (cached).",
    "2. RoBERTa-D inference: Debt or credit management (0.47)",
    "3. ModernBERT inference: Debt collection (0.52)",
    "4. LR Stacker decision: Debt or credit management (0.40)",
    "5. Thompson Sampling: Arm 3 (Human Escalation) selected",
    "6. RAG retrieved 3 passages for 'Debt or credit management'",
  ],
};
