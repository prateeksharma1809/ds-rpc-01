import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def format_context_with_metadata(docs: List[Dict[str, Any]], include_scores: bool = True) -> str:
    """Format context with enhanced metadata for better RAG performance."""
    formatted_chunks = []
    
    for i, doc in enumerate(docs, 1):
        # Base content
        content = doc.get("content", "")
        
        # Enhanced metadata
        section_title = doc.get("section_title", "Unknown Section")
        source = doc.get("source", "Unknown Source")
        role = doc.get("role", "general")
        file_type = doc.get("file_type", "")
        word_count = doc.get("word_count", 0)
        chunk_index = doc.get("chunk_index", 0)
        total_chunks = doc.get("total_chunks", -1)
        score = doc.get("score", 0.0)
        
        # Format the chunk with metadata
        chunk_header = f"[Document {i}: {section_title}]"
        if include_scores:
            chunk_header += f" (Relevance: {score:.3f})"
        
        chunk_header += f"\nSource: {source}"
        chunk_header += f" | Role: {role}"
        chunk_header += f" | Type: {file_type}"
        
        if total_chunks > 0:
            chunk_header += f" | Part: {chunk_index + 1}/{total_chunks}"
        
        chunk_header += f" | Words: {word_count}"
        
        # Add structured data for CSV files
        row_data = doc.get("row_data", {})
        if row_data:
            chunk_header += f"\nStructured Data: {json.dumps(row_data, indent=2)}"
        
        formatted_chunk = f"{chunk_header}\n\n{content}\n"
        formatted_chunks.append(formatted_chunk)
    
    return "\n" + "="*80 + "\n".join(formatted_chunks)


def build_role_aware_prompt(query: str, docs: List[Dict[str, Any]], user_role: str) -> str:
    """Build a role-aware prompt that considers the user's role and document context."""
    
    # Analyze document types and roles
    doc_roles = [doc.get("role", "general") for doc in docs]
    doc_types = [doc.get("file_type", "") for doc in docs]
    has_csv_data = any(doc.get("row_data") for doc in docs)
    
    # Determine context type
    context_type = "mixed"
    if all(role == "general" for role in doc_roles):
        context_type = "general"
    elif len(set(doc_roles)) == 1:
        context_type = f"{doc_roles[0]}-specific"
    
    # Build role-specific instructions
    role_instructions = {
        "engineering": "Focus on technical details, architecture, and implementation specifics. Use technical terminology appropriately.",
        "marketing": "Emphasize business impact, customer benefits, and strategic insights. Use marketing-friendly language.",
        "finance": "Highlight financial metrics, costs, ROI, and quantitative analysis. Be precise with numbers and percentages.",
        "hr": "Focus on people-related aspects, employee data, and organizational information. Respect privacy and confidentiality.",
        "c-level": "Provide high-level strategic insights and executive summaries. Focus on business impact and key decisions.",
        "general": "Provide balanced, comprehensive information suitable for a general audience."
    }
    
    role_instruction = role_instructions.get(user_role, role_instructions["general"])
    
    # Build content-specific instructions
    content_instructions = []
    if has_csv_data:
        content_instructions.append("- When referencing CSV data, cite specific field names and values clearly.")
    
    if "md" in doc_types:
        content_instructions.append("- When referencing markdown documents, cite the specific section titles.")
    
    content_instruction = "\n".join(content_instructions) if content_instructions else ""
    
    # Build the enhanced prompt
    context = format_context_with_metadata(docs, include_scores=True)
    
    prompt = f"""You are a helpful assistant at a FinTech company, specifically tailored to help {user_role} users.

{role_instruction}

Answer the following question based on the provided context. Be precise and reference the content, but do not hallucinate facts. If you are unsure, say so.

Guidelines:
- Reference specific sections and sources when possible
- Use the relevance scores to prioritize more relevant information
- Provide clear, actionable insights based on the user's role
- If the context contains structured data (CSV), reference specific fields and values
- Maintain appropriate tone and detail level for the user's role{content_instruction}

Context:
{context}

Question: {query}

Answer:"""
    
    return prompt


def build_simple_prompt(query: str, docs: List[Dict[str, Any]]) -> str:
    """Build a simple prompt for backward compatibility."""
    context_chunks = [
        f"[{doc['section_title']} - Level {doc['heading_level']}]\n{doc['content']} - \n Source: {doc['source']}"
        for doc in docs
    ]
    context = "\n\n".join(context_chunks)
    
    return f"""You are a helpful assistant at a FinTech company.

Answer the following question based on the provided context. Be precise and reference the content, but do not hallucinate facts. If you are unsure, say so.
Answer should be clear and insightful. Also providing reference to the source document.

Context:
{context}

Question:
{query}

Answer:"""


def generate_response(
    query: str, 
    docs: List[Dict[str, Any]], 
    user_role: str = "general",
    use_enhanced_prompt: bool = True,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    max_chunks: int = 1  # Default
) -> str:
    """
    Generate a response using the RAG engine with enhanced capabilities.
    
    Args:
        query: The user's question
        docs: List of document chunks with metadata
        user_role: The user's role for role-aware responses
        use_enhanced_prompt: Whether to use the enhanced prompt or simple prompt
        temperature: Controls randomness in response generation
        max_tokens: Maximum tokens for the response
    """
    
    if not docs:
        return "I don't have enough information to answer your question. Please try rephrasing or ask about a different topic."
    
    has_csv_data = any(doc.get("row_data") for doc in docs)
    if has_csv_data:
        max_chunks = max(max_chunks, 5)  # Use at least 5 chunks if CSV data is present

    # Limit the number of chunks to avoid exceeding token limits
    docs = docs[:max_chunks]

    # Choose prompt type
    if use_enhanced_prompt:
        prompt = build_role_aware_prompt(query, docs, user_role)
    else:
        prompt = build_simple_prompt(query, docs)
    print(prompt)
    
    # Prepare API call parameters
    api_params = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    
    if max_tokens:
        api_params["max_tokens"] = max_tokens
    
    try:
        response = client.chat.completions.create(**api_params)
        return response.choices[0].message.content
    except Exception as e:
        return f"I encountered an error while generating a response: {str(e)}. Please try again."


def generate_structured_response(
    query: str, 
    docs: List[Dict[str, Any]], 
    user_role: str = "general"
) -> Dict[str, Any]:
    """
    Generate a structured response with additional metadata.
    
    Returns:
        Dictionary containing response, sources, and metadata
    """
    
    # Generate the main response
    response = generate_response(query, docs, user_role, use_enhanced_prompt=True)
    
    # Extract source information
    sources = []
    for doc in docs:
        source_info = {
            "title": doc.get("section_title", "Unknown"),
            "source": doc.get("source", "Unknown"),
            "role": doc.get("role", "general"),
            "score": doc.get("score", 0.0),
            "file_type": doc.get("file_type", ""),
            "word_count": doc.get("word_count", 0)
        }
        
        # Add structured data for CSV files
        if doc.get("row_data"):
            source_info["structured_data"] = doc["row_data"]
        
        sources.append(source_info)
    
    # Calculate response metadata
    total_words = sum(doc.get("word_count", 0) for doc in docs)
    avg_score = sum(doc.get("score", 0) for doc in docs) / len(docs) if docs else 0
    role_distribution = {}
    for doc in docs:
        role = doc.get("role", "general")
        role_distribution[role] = role_distribution.get(role, 0) + 1
    
    return {
        "response": response,
        "sources": sources,
        "metadata": {
            "total_sources": len(docs),
            "total_words_referenced": total_words,
            "average_relevance_score": round(avg_score, 3),
            "role_distribution": role_distribution,
            "user_role": user_role,
            "query": query
        }
    }

