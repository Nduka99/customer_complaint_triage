// Example complaints shown in the app. `expect` records what the deployed model was measured to
// do with each one; scripts/check-examples.mjs re-checks it against the live backend, so a label
// can never claim an outcome the model does not produce.
//   "automatic": clears both confidence gates.
//   "escalate":  sent to a person by a confidence gate (not by the routing's random draw).
export const EXAMPLES = [
  {
    label: "Hidden bank fees",
    text: "I opened a checking account and was charged hidden fees that were not disclosed during signup. The bank said these were standard maintenance fees but they were never mentioned in any of the paperwork I signed.",
    expect: "automatic",
  },
  {
    label: "Credit report error",
    text: "My credit report shows a late payment that I never made. I have proof of on-time payment including bank statements and receipts. I have disputed this with the credit bureau twice but nothing has changed.",
    expect: "automatic",
  },
  {
    label: "Debt collector harassment",
    text: "A debt collector keeps calling me about a debt I already paid off and is threatening legal action. I have sent them proof of payment three times but they refuse to acknowledge it and continue to call daily.",
    expect: "automatic",
  },
  {
    label: "Identity theft",
    text: "Someone opened a credit card in my name; the bank closed it but it still shows on my credit report and a collection agency is chasing me.",
    expect: "escalate",
    why: "Spans a card, a credit report and a debt collector",
  },
  {
    label: "Payday loan withdrawals",
    text: "A payday lender took repeated withdrawals from my prepaid card and now my bank account is overdrawn.",
    expect: "escalate",
    why: "Spans a payday loan, a prepaid card and a bank account",
  },
];
