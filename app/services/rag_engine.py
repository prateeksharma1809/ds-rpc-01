import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def build_prompt(query: str, documents: list[str]) -> str:
    context = "\n\n".join(documents)
    return f"""You are a helpful assistant at a FinTech company.

Answer the following question based on the provided context. Be precise and reference the content, but do not hallucinate facts. If you are unsure, say so.
Answer should be clear and insightful. Also providing reference to the source document.


Context:
{context}

Question:
{query}

Answer:"""

def generate_response(query: str, docs: list[dict]) -> str:
    context_chunks = [
        f"[{doc['section_title']} - Level {doc['heading_level']}]\n{doc['content']} - \n Source: {doc['source']}"
        for doc in docs
    ]
    prompt = build_prompt(query, context_chunks)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content
