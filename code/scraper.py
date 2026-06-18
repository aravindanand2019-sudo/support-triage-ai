import json
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MAX_PAGES_PER_DOMAIN = 60
MAX_DEPTH = 2
REQUEST_DELAY_SECONDS = 0.5
TIMEOUT_SECONDS = 12
MAX_FETCH_ATTEMPTS = 3

START_URLS = {
    "hackerrank": "https://support.hackerrank.com/hc/en-us",
    "claude": "https://support.claude.com/en/",
    "visa": "https://www.visa.co.in/support.html",
}

OUTPUT_FILES = {
    "hackerrank": DATA_DIR / "hackerrank_corpus.json",
    "claude": DATA_DIR / "claude_corpus.json",
    "visa": DATA_DIR / "visa_corpus.json",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 support-triage-hackathon-corpus-builder/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def normalize_url(url):
    clean_url, _ = urldefrag(url)
    parsed = urlparse(clean_url)
    if parsed.scheme not in ("http", "https"):
        return ""
    return clean_url.rstrip("/")


def same_domain(url, base_netloc):
    return urlparse(url).netloc.lower() == base_netloc.lower()


def looks_like_html_page(url):
    lower_url = url.lower()
    blocked_suffixes = (
        ".css",
        ".js",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".mp4",
        ".mp3",
        ".woff",
        ".woff2",
    )
    for suffix in blocked_suffixes:
        if lower_url.endswith(suffix):
            return False
    return True


def clean_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "svg", "form"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    return title, text


def extract_links(html, current_url, base_netloc):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = normalize_url(urljoin(current_url, href))
        if not absolute:
            continue
        if same_domain(absolute, base_netloc) and looks_like_html_page(absolute):
            links.append(absolute)
    return links


def fetch_page(session, url):
    last_error = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            content_type = response.headers.get("content-type", "").lower()
            if response.status_code >= 400:
                print(f"  skipped {url} status={response.status_code}")
                return None
            if "text/html" not in content_type and "application/xhtml" not in content_type and content_type:
                print(f"  skipped {url} content-type={content_type}")
                return None
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_FETCH_ATTEMPTS:
                print(f"  retrying {url} after error={exc}")
                time.sleep(REQUEST_DELAY_SECONDS)
            else:
                print(f"  skipped {url} error={last_error}")
    return None


def scrape_domain(domain, start_url):
    print(f"\nScraping {domain}: {start_url}")
    base_netloc = urlparse(start_url).netloc
    queue = deque([(normalize_url(start_url), 0)])
    visited = set()
    chunks = []

    with requests.Session() as session:
        while queue and len(visited) < MAX_PAGES_PER_DOMAIN:
            url, depth = queue.popleft()
            if not url or url in visited:
                continue
            if not same_domain(url, base_netloc):
                continue

            visited.add(url)
            print(f"[{domain}] {len(visited):02d}/{MAX_PAGES_PER_DOMAIN} depth={depth} {url}")
            html = fetch_page(session, url)
            time.sleep(REQUEST_DELAY_SECONDS)
            if not html:
                continue

            title, text = clean_text_from_html(html)
            if len(text.split()) >= 20:
                chunks.append(
                    {
                        "domain": domain,
                        "url": url,
                        "title": title,
                        "text": text,
                    }
                )

            if depth >= MAX_DEPTH:
                continue

            for link in extract_links(html, url, base_netloc):
                if link not in visited:
                    queue.append((link, depth + 1))

    return chunks


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for domain, start_url in START_URLS.items():
        chunks = scrape_domain(domain, start_url)
        output_path = OUTPUT_FILES[domain]
        if not chunks and output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = []
            if existing:
                print(f"Keeping existing {len(existing)} chunks in {output_path}; fresh scrape was empty.")
                continue
        output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()
