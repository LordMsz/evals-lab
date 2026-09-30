"""Tiny knowledge base + naive keyword retriever (stand-in for your vector store)."""
import os
import re

DOCS = {
    "KB-REFUND": "Refund policy: purchases can be refunded within 30 days. Refunds go back to the "
                 "original payment method within 5-7 business days. Duplicate charges are always "
                 "refunded in full once confirmed by the billing team.",
    "KB-CANCEL": "Cancellation: you can cancel your subscription at any time in Settings > Billing. "
                 "Cancellation takes effect at the end of the current billing period. We do not "
                 "offer partial or prorated refunds for unused time.",
    "KB-PRICING": "Pricing: Basic plan costs 9 EUR per month, Pro plan costs 29 EUR per month. "
                  "Annual billing gives 2 months free (you pay for 10 months).",
    "KB-SHIPPING": "Shipping: standard delivery takes 3-5 business days, express delivery takes "
                   "1-2 business days. We ship to EU countries, the UK and Switzerland only.",
    "KB-LOST": "Late or lost packages: if a package has not arrived 2 business days after the expected "
               "delivery date, support opens a carrier claim and offers a replacement or refund.",
    "KB-PASSWORD": "Password reset: request a reset link on the login page. The link is valid for "
                   "30 minutes and can be used once. If it expired, request a new link.",
    "KB-2FA": "Two-factor lockout: if you lost access to your 2FA device, use a backup code. Without "
              "backup codes, contact support; identity verification with a photo ID is required. "
              "Account recovery takes up to 48 hours.",
    "KB-RATELIMIT": "API rate limits: Basic plan allows 60 requests per minute, Pro plan allows 600 "
                    "requests per minute. Exceeding the limit returns HTTP 429; retry with backoff.",
    "KB-EXPORT": "Data export: export all data as CSV or JSON in Settings > Data > Export. "
                 "Large exports can take up to 24 hours and are delivered by email.",
}

_STOP = set("the a an and or to of in on for is are be it i my me you your we our this that with "
            "can do does how what when why where will was at by from not no".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2}


# Found by the eval suite: "charged twice" never matched "Duplicate charges" (lexical gap).
# RETRIEVER=naive reproduces the original bug.
_SYNONYMS = {"twice": {"duplicate"}, "charged": {"charges", "duplicate"}, "money": {"refund", "refunds"},
             "back": {"refund"}, "locked": {"lockout"}}


def retrieve(query: str, k: int = 2) -> list[dict]:
    q = _tokens(query)
    if os.getenv("RETRIEVER", "synonyms") != "naive":
        q |= {s for t in q for s in _SYNONYMS.get(t, ())}
    scored = []
    for doc_id, text in DOCS.items():
        overlap = len(q & _tokens(text))
        if overlap:
            scored.append((overlap, doc_id, text))
    scored.sort(reverse=True)
    return [{"id": d, "text": t, "score": s} for s, d, t in scored[:k]]
