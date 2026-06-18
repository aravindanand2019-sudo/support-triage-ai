from groq import Groq
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY")) if os.environ.get("GROQ_API_KEY") else None


def call_llm(system_prompt, user_message):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=400,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = (
    "You are a support agent. Answer ONLY using the provided context.\n"
    "Do not add any information not present in the context.\n"
    "If the context is insufficient, say so and recommend contacting support directly.\n"
    "Be concise, empathetic, and professional. Max 3 sentences."
)

SUPPORT_URLS = {
    "hackerrank": "https://support.hackerrank.com/hc/en-us/requests/new",
    "claude": "https://support.claude.com/en/",
    "visa": "https://www.visa.co.in/support.html",
    "unknown": "the relevant support team",
    "general": "the relevant support team",
}


def _context_text(context_chunks):
    lines = []
    for chunk in context_chunks:
        area = chunk.get("product_area", "general_support")
        text = chunk.get("text", "")
        if len(text) > 1600:
            text = text[:1600].rstrip() + "..."
        lines.append(f"[{chunk.get('domain', 'general')} / {area}] {text}")
    return "\n\n".join(lines)


def _fallback_reply(issue, subject, context_chunks):
    if not context_chunks:
        return "I do not have enough support documentation to answer this safely. Please contact support directly."
    top_chunk = context_chunks[0]
    title = top_chunk.get("title") or top_chunk.get("product_area", "support documentation")
    domain = top_chunk.get("domain", "the product")
    return (
        f"I found relevant {domain} support documentation in {title}, but I could not generate a live answer from the LLM. "
        "Please review that documentation or contact support directly as the next step."
    )


def generate_reply(issue, subject, context_chunks):
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return _fallback_reply(issue, subject, context_chunks)

    system_prompt = (
        "You are a support agent. Use ONLY the context below to answer.\n"
        "Do not copy the context - synthesize a helpful answer in your own words.\n"
        "Be empathetic, concise, max 3 sentences. End with one clear next step."
    )
    user_prompt = (
        f"Customer issue: {issue or ''}\n\n"
        "Context from support documentation:\n"
        f"{_context_text(context_chunks)}\n\n"
        "Write a helpful support response:"
    )

    try:
        response = call_llm(system_prompt, user_prompt).strip()
        if response:
            return response
    except Exception as exc:
        return f"I do not have enough support documentation to answer this safely. Please contact support directly. API response generation failed: {exc}"

    return _fallback_reply(issue, subject, context_chunks)


def generate_justification(issue, status, top_chunk, escalation_reason):
    title = ""
    url = ""
    text = ""
    product_area = "general_support"
    domain = "general"
    if top_chunk:
        title = top_chunk.get("title", "")
        url = top_chunk.get("url", "")
        text = top_chunk.get("text", "")
        product_area = top_chunk.get("product_area", "general_support")
        domain = top_chunk.get("domain", "general")

    fallback_source = title or url or f"{domain} {product_area} documentation"
    if not os.environ.get("GROQ_API_KEY", "").strip():
        if status == "Escalated":
            return f"Escalated because {escalation_reason}, supported by {fallback_source}."
        return f"Replied because {fallback_source} contains documentation relevant to the user's issue."

    system_prompt = (
        "You write concise support triage justifications. Use ONLY the provided documentation. "
        "Do not copy the documentation text. Write exactly one sentence explaining why the chosen status was selected "
        "and what specific documentation supported it."
    )
    user_message = (
        f"Customer issue: {issue or ''}\n\n"
        f"Chosen status: {status}\n"
        f"Escalation reason if any: {escalation_reason or 'None'}\n\n"
        "Context from top retrieved support documentation:\n"
        f"Title: {title}\n"
        f"URL: {url}\n"
        f"Product area: {product_area}\n"
        f"Text: {text[:1800]}\n\n"
        "Write the one-sentence justification:"
    )

    try:
        response = call_llm(system_prompt, user_message).strip()
        if response:
            return " ".join(response.split())
    except Exception:
        pass

    if status == "Escalated":
        return f"Escalated because {escalation_reason}, supported by {fallback_source}."
    return f"Replied because {fallback_source} contains documentation relevant to the user's issue."


def generate_escalation(issue, domain, reason):
    target = SUPPORT_URLS.get(domain, SUPPORT_URLS["unknown"])
    if target.startswith("http"):
        return f"Thanks for sharing this. This issue needs human review because {reason}. Please contact support here: {target}"
    return f"Thanks for sharing this. This issue needs human review because {reason}. Please contact {target}."
