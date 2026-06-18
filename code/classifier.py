import re


DOMAIN_KEYWORDS = {
    "hackerrank": [
        "hackerrank",
        "assessment",
        "test",
        "candidate",
        "recruiter",
        "plagiarism",
        "proctor",
        "score",
        "invite",
        "variant",
    ],
    "claude": [
        "claude",
        "anthropic",
        "conversation",
        "chat",
        "pro",
        "api",
        "project",
        "memory",
        "model",
        "message limit",
    ],
    "visa": [
        "visa",
        "card",
        "transaction",
        "chargeback",
        "dispute",
        "pin",
        "atm",
        "stolen card",
        "traveller",
        "cheque",
    ],
}


BUG_KEYWORDS = ["error", "broken", "not working", "crash", "glitch", "bug", "fail", "down", "inaccessible"]
FEATURE_KEYWORDS = ["would like", "can you add", "feature", "suggestion", "wish", "request"]
INVALID_POSITIVE = ["thank you", "thanks", "appreciate", "helping me"]
OFF_TOPIC_KEYWORDS = [
    "actor",
    "iron man",
    "movie",
    "weather",
    "recipe",
    "delete all files",
    "all files from the system",
]
SENSITIVE_KEYWORDS = [
    "fraud",
    "unauthorized",
    "stolen",
    "hacked",
    "suspicious activity",
    "legal",
    "lawsuit",
    "complaint",
    "account suspended",
    "permanently banned",
]
VISA_DISPUTE_KEYWORDS = [
    "wrong product",
    "merchant ignoring",
    "seller ignoring",
    "not received",
    "item not arrived",
]


def _normalize(text):
    return (text or "").strip().lower()


def get_domain(company_col, issue_text):
    company = _normalize(company_col)
    if company and company not in ("none", "nan", "null"):
        if "hackerrank" in company:
            return "hackerrank"
        if "claude" in company or "anthropic" in company:
            return "claude"
        if "visa" in company:
            return "visa"

    text = _normalize(issue_text)
    best_domain = "unknown"
    best_score = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def get_product_area(retrieved_chunks):
    if not retrieved_chunks:
        return ""
    return retrieved_chunks[0].get("product_area", "")


def get_request_type(issue_text):
    text = _normalize(issue_text)
    if not text:
        return "invalid"

    if len(text) < 15:
        return "invalid"

    words = re.findall(r"[a-zA-Z]{2,}", text)
    if len(words) < 3:
        return "invalid"

    for keyword in OFF_TOPIC_KEYWORDS:
        if keyword in text:
            return "invalid"

    if len(words) <= 6:
        for phrase in INVALID_POSITIVE:
            if phrase in text:
                return "invalid"

    for phrase in INVALID_POSITIVE:
        if phrase == text:
            return "invalid"

    for keyword in BUG_KEYWORDS:
        if keyword in text:
            return "bug"

    for keyword in FEATURE_KEYWORDS:
        if keyword in text:
            return "feature_request"

    return "product_issue"


def should_escalate(domain, issue_text, retrieved_chunks):
    text = _normalize(issue_text)
    request_type = get_request_type(issue_text)

    if request_type == "invalid":
        return False, ""

    if request_type == "bug" and domain == "unknown":
        return True, "unidentified outage or unsupported bug report requires human review"

    for keyword in SENSITIVE_KEYWORDS:
        if keyword in text:
            if domain == "visa" and keyword == "stolen" and "identity" in text:
                return True, "identity theft requires human review"
            if domain == "visa" and keyword == "stolen":
                return False, ""
            return True, "sensitive or high-risk issue requires human review"

    if "chargeback" in text or "dispute" in text:
        if domain == "visa":
            return True, "Visa dispute or chargeback requires issuer or human review"

    if domain == "visa":
        for keyword in VISA_DISPUTE_KEYWORDS:
            if keyword in text:
                return True, "Visa merchant dispute requires human handling"
        if "refund" in text:
            return True, "Visa refund or dispute request requires human handling"

    top_score = 0.0
    if retrieved_chunks:
        top_score = retrieved_chunks[0].get("score", 0.0)
    if top_score < 0.05:
        return True, "insufficient corpus coverage"

    for chunk in retrieved_chunks[:2]:
        if chunk.get("escalate_if_matched") and chunk.get("score", 0.0) >= 0.12:
            return True, "matched documentation indicates human review is needed"

    return False, ""
