import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def infer_product_area(entry):
    domain = entry.get("domain", "")
    text = f"{entry.get('title', '')} {entry.get('url', '')} {entry.get('text', '')}".lower()

    if "privacy" in text or "gdpr" in text or "personal data" in text or "private" in text:
        return "privacy"
    if "conversation" in text or "chat" in text or "project" in text or "rename" in text:
        return "conversation_management"
    if domain == "hackerrank" and ("community" in text or "delete account" in text or "profile" in text):
        return "community"
    if domain == "visa" and (
        "traveller" in text or "traveler" in text or "travel" in text or "international" in text
    ):
        return "travel_support"
    if domain == "hackerrank" and (
        "test" in text
        or "assessment" in text
        or "candidate" in text
        or "login" in text
        or "password" in text
        or "score" in text
        or "proctor" in text
        or "invite" in text
        or "interview" in text
    ):
        return "screen"
    return "general_support"


def load_corpus(corpus_dir=DATA_DIR):
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.json")):
        try:
            raw_items = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping corpus file {path}: {exc}")
            continue

        if not isinstance(raw_items, list):
            continue

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            doc = {
                "domain": str(item.get("domain", "general")).strip().lower() or "general",
                "url": str(item.get("url", "")),
                "title": str(item.get("title", "")),
                "text": text,
                "product_area": str(item.get("product_area", "")).strip(),
                "escalate_if_matched": bool(item.get("escalate_if_matched", False)),
            }
            if not doc["product_area"]:
                doc["product_area"] = infer_product_area(doc)
            docs.append(doc)

    if not docs:
        raise ValueError("No corpus documents found in data/*.json. Run python code/scraper.py first.")

    return docs


class Retriever:
    def __init__(self, corpus_dir=DATA_DIR):
        self.knowledge_base = load_corpus(corpus_dir)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
            stop_words="english",
        )
        texts = [entry["text"] for entry in self.knowledge_base]
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query, top_k=5, domain_filter=None):
        if not query or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        results = []

        index = 0
        for entry in self.knowledge_base:
            if domain_filter and entry["domain"] not in (domain_filter, "general"):
                index += 1
                continue
            item = dict(entry)
            item["score"] = float(scores[index])
            results.append(item)
            index += 1

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
