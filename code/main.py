from dotenv import load_dotenv

load_dotenv()

import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from classifier import get_domain, get_product_area, get_request_type, should_escalate
from responder import generate_escalation, generate_justification, generate_reply
from retriever import Retriever


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH = ROOT / "support_tickets" / "output.csv"
LOG_PATH = ROOT / "logs" / "log.txt"


def _safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def _find_column(columns, desired):
    desired_lower = desired.lower()
    for column in columns:
        if column.lower() == desired_lower:
            return column
    return None


def _write_log(message):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(message + "\n")


def _log_ticket(index, issue, subject, company, domain, request_type, product_area, status, reason, chunks, response):
    _write_log(f"## {datetime.now().isoformat(timespec='seconds')} Ticket {index}")
    _write_log(f"Company: {company}")
    _write_log(f"Subject: {subject}")
    _write_log(f"Issue: {issue}")
    _write_log(f"Domain: {domain}")
    _write_log(f"Request Type: {request_type}")
    _write_log(f"Product Area: {product_area}")
    _write_log(f"Status: {status}")
    _write_log(f"Escalation Reason: {reason}")
    _write_log("Retrieved:")
    for chunk in chunks:
        _write_log(
            f"- {chunk.get('domain')} / {chunk.get('product_area')} / "
            f"{chunk.get('score', 0.0):.4f}: {chunk.get('text', '')[:300]}"
        )
    _write_log(f"Response: {response}")
    _write_log("")


def _domain_filter(domain):
    if domain in ("hackerrank", "claude", "visa"):
        return domain
    return None


def _is_short_polite_closing(issue):
    text = issue.lower().strip()
    words = text.split()
    polite_phrases = ["thank you", "thanks", "happy to help"]
    if len(words) > 6:
        return False
    for phrase in polite_phrases:
        if phrase in text:
            return True
    return False


def _normalize_product_area(domain, issue, request_type, product_area, escalated):
    text = issue.lower()
    if request_type == "invalid":
        if _is_short_polite_closing(issue):
            return ""
        return "conversation_management"
    if escalated and domain == "unknown":
        return ""
    if domain == "hackerrank" and "delete" in text and "account" in text:
        return "community"
    return product_area


def run():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        f"Support triage run started {datetime.now().isoformat(timespec='seconds')}\n\n",
        encoding="utf-8",
    )

    retriever = Retriever()
    frame = pd.read_csv(INPUT_PATH)
    issue_col = _find_column(frame.columns, "issue")
    subject_col = _find_column(frame.columns, "subject")
    company_col = _find_column(frame.columns, "company")

    if not issue_col:
        raise ValueError("Could not find required issue column")
    if not subject_col:
        raise ValueError("Could not find required subject column")
    if not company_col:
        raise ValueError("Could not find required company column")

    output_rows = []
    row_number = 1
    for _, row in frame.iterrows():
        issue = _safe_text(row[issue_col])
        subject = _safe_text(row[subject_col])
        company = _safe_text(row[company_col])
        combined_query = f"{subject} {issue}".strip()

        domain = get_domain(company, issue)
        chunks = retriever.retrieve(combined_query, top_k=5, domain_filter=_domain_filter(domain))
        request_type = get_request_type(issue)
        escalated, reason = should_escalate(domain, issue, chunks)
        product_area = get_product_area(chunks)

        if not product_area:
            product_area = "conversation_management" if request_type == "invalid" else "general_support"

        if escalated:
            status = "Escalated"
            response = generate_escalation(issue, domain, reason)
        else:
            status = "Replied"
            if request_type == "invalid":
                response = "I am sorry, this is out of scope from my capabilities"
            else:
                response = generate_reply(issue, subject, chunks)

        product_area = _normalize_product_area(domain, issue, request_type, product_area, escalated)
        top_chunk = chunks[0] if chunks else {}
        justification = generate_justification(issue, status, top_chunk, reason)
        output_rows.append(
            {
                "Issue": issue,
                "Subject": subject,
                "Company": company,
                "status": status,
                "product_area": product_area,
                "response": response,
                "justification": justification,
                "request_type": request_type,
            }
        )

        _log_ticket(
            row_number,
            issue,
            subject,
            company,
            domain,
            request_type,
            product_area,
            status,
            reason,
            chunks,
            response,
        )

        print(f"Ticket {row_number}: domain={domain} status={status} request_type={request_type}")
        time.sleep(2)
        row_number += 1

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Issue",
                "Subject",
                "Company",
                "status",
                "product_area",
                "response",
                "justification",
                "request_type",
            ],
        )
        writer.writeheader()
        for output_row in output_rows:
            writer.writerow(output_row)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
