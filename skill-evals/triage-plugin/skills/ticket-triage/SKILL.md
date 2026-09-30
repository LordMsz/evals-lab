---
name: ticket-triage
description: Triage a customer support ticket for the ACME subscription product into category and priority and draft a policy-compliant reply. Use whenever the user pastes a customer message, support ticket or complaint and wants it classified, prioritized or answered.
---

# Ticket triage

Classify the ticket and draft a reply that follows ACME policy exactly.

## Policy (the only facts you may state)
- Refunds: within 30 days, back to the original payment method in 5-7 business days. Duplicate charges are refunded in full once billing confirms.
- Cancellation: any time in Settings > Billing, effective at the end of the billing period. **No partial or prorated refunds for unused time.**
- Shipping: EU, UK and Switzerland only. Standard 3-5 business days, express 1-2.
- Never promise credits, refunds or dates that are not stated above. If unsure, say a human agent will follow up.

## Output format
Reply with exactly this block and nothing else:

```
TRIAGE
category: <billing|shipping|account|technical|other>
priority: <low|normal|high|urgent>
reply: <max 3 sentences>
```
