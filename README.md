# Multi-Domain Support Triage AI Agent 🚀

A high-performance, RAG-lite support triage agent built for the **HackerRank Orchestrate Hackathon**. This agent intelligently routes, classifies, and responds to customer support tickets across three distinct domains: **HackerRank**, **Anthropic (Claude)**, and **Visa**.

## 🌟 Key Features
- **Intelligent Domain Routing:** Automatically detects whether a ticket belongs to Visa, HackerRank, or Claude based on content and metadata.
- **Safety-First Escalation:** Built-in rule engine to identify high-risk scenarios (Fraud, Identity Theft, Legal disputes) and escalate them to human agents instantly.
- **Grounded AI Responses:** Uses **Groq (LLaMA 3.3 70B)** to generate responses strictly grounded in the provided support corpus to prevent hallucinations.
- **Hybrid RAG Pipeline:** Combines TF-IDF vector retrieval with rule-based classification for high accuracy and low latency.

## 🛠️ Tech Stack
- **LLM:** LLaMA 3.3 70B (via Groq API)
- **Language:** Python 3.x
- **Libraries:** Scikit-learn (TF-IDF), Pandas, NumPy, BeautifulSoup4
- **Orchestration:** Custom-built modular pipeline

## 🏗️ Architecture
- `code/classifier.py`: Domain detection, request typing, and escalation logic.
- `code/retriever.py`: TF-IDF indexing and cosine similarity search for knowledge retrieval.
- `code/responder.py`: LLM prompt engineering and response generation.
- `code/main.py`: Central orchestrator for batch processing support tickets.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- A [Groq API Key](https://console.groq.com/)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/aravindanand2019-sudo/support-triage-ai.git
cd support-triage-ai

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_actual_key_here
```

### 4. Usage
```bash
python code/main.py
```

## 📊 Sample Output
The agent processes `support_issues.csv` and generates a detailed `output.csv` containing:
- **Status:** (Replied / Escalated)
- **Product Area:** (e.g., Billing, Technical, Account Access)
- **Response:** The user-facing answer.
- **Justification:** Why the agent made that specific decision.
- **Request Type:** (Bug, Feature Request, Product Issue, Invalid)

---
*Created for the HackerRank Orchestrate Hackathon (May 2026).*
