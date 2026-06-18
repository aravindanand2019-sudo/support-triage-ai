# Multi-Domain Support Triage Agent

## Setup
1. pip install -r requirements.txt
2. Create .env file with: GROQ_API_KEY=your-key-here
3. python code/main.py

## Approach
- TF-IDF bigrams (unigrams + bigrams) over scraped support corpus
- Rule-based escalation for fraud, disputes, identity theft, legal issues
- Groq LLaMA 3.3 70B for grounded response generation
- Corpus scraped from HackerRank, Claude, and Visa support sites

## Architecture
- code/corpus_loader.py  → loads data/ corpus chunks
- code/retriever.py      → TF-IDF index + cosine similarity search
- code/classifier.py     → domain, issue type, escalation rules
- code/responder.py      → Groq API response generation
- code/main.py           → pipeline orchestrator

## Escalation Rules
Always escalates: fraud, unauthorized transactions, identity theft,
chargebacks, legal mentions, retrieval confidence below threshold

## Output
support_tickets/output.csv with columns:
status, product_area, response, justification, request_type
