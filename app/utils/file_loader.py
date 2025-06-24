import re
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special markdown characters that don't add meaning
    text = re.sub(r'[*_`~]', '', text)
    # Clean up table formatting
    text = re.sub(r'\|+', ' | ', text)
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def extract_headers_and_content(text: str) -> List[Dict[str, Any]]:
    """Extract headers and their associated content from markdown text."""
    lines = text.split('\n')
    sections = []
    current_section = None
    current_content = []
    
    for line in lines:
        # Check if line is a header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        
        if header_match:
            # Save previous section if exists
            if current_section and current_content:
                sections.append({
                    'header': current_section['header'],
                    'level': current_section['level'],
                    'content': '\n'.join(current_content).strip()
                })
            
            # Start new section
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            current_section = {
                'header': header_text,
                'level': level
            }
            current_content = []
        else:
            # Add line to current content
            if current_section is not None:
                current_content.append(line)
    
    # Add the last section
    if current_section and current_content:
        sections.append({
            'header': current_section['header'],
            'level': current_section['level'],
            'content': '\n'.join(current_content).strip()
        })
    
    return sections


def chunk_content(content: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split content into overlapping chunks for better retrieval."""
    words = content.split()
    chunks = []
    
    if len(words) <= max_chunk_size:
        return [content]
    
    for i in range(0, len(words), max_chunk_size - overlap):
        chunk_words = words[i:i + max_chunk_size]
        chunk_text = ' '.join(chunk_words)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks


def load_markdown(file_path: str, max_chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Load and process markdown files with improved parsing."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    # Clean the text
    text = clean_text(text)
    
    # Extract sections with headers
    sections = extract_headers_and_content(text)
    
    chunks = []
    
    if sections:
        # Process structured markdown with headers
        for section in sections:
            if not section['content'].strip():
                continue
                
            # Split section content into chunks
            section_chunks = chunk_content(section['content'], max_chunk_size, overlap)
            
            for i, chunk in enumerate(section_chunks):
                if len(chunk.split()) < 10:  # Skip very short chunks
                    continue
                    
                chunks.append({
                    "content": chunk,
                    "section_title": section['header'],
                    "heading_level": section['level'],
                    "chunk_index": i,
                    "total_chunks": len(section_chunks)
                })
    else:
        # Fallback for documents without clear headers
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = []
        chunk_index = 0
        
        for para in paragraphs:
            current_chunk.append(para)
            
            # Create chunk when we have enough content
            if len(' '.join(current_chunk).split()) >= max_chunk_size:
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text.split()) >= 10:
                    chunks.append({
                        "content": chunk_text,
                        "section_title": "Document Content",
                        "heading_level": 1,
                        "chunk_index": chunk_index,
                        "total_chunks": -1  # Unknown total
                    })
                    chunk_index += 1
                
                # Keep some overlap
                overlap_paras = current_chunk[-1:] if len(current_chunk) > 1 else []
                current_chunk = overlap_paras
        
        # Add remaining content
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text.split()) >= 10:
                chunks.append({
                    "content": chunk_text,
                    "section_title": "Document Content",
                    "heading_level": 1,
                    "chunk_index": chunk_index,
                    "total_chunks": -1
                })
    
    return chunks


def load_csv(file_path: str, max_chunk_size: int = 300) -> List[Dict[str, Any]]:
    """Load and process CSV files with improved structure preservation."""
    df = pd.read_csv(file_path)
    
    chunks = []
    
    # Get column names for context
    columns = df.columns.tolist()
    
    # Process each row with meaningful context
    for index, row in df.iterrows():
        # Create a structured representation of the row
        row_data = {}
        for col in columns:
            value = str(row[col]).strip()
            if value and value != 'nan':
                row_data[col] = value
        
        if not row_data:
            continue
        
        # Create meaningful content from the row
        content_parts = []
        for col, value in row_data.items():
            content_parts.append(f"{col}: {value}")
        
        content = " | ".join(content_parts)
        
        # Skip if content is too short
        if len(content.split()) < 5:
            continue
        
        # Create section title based on the most important field
        # Try to find a name or ID field for better context
        name_fields = ['name', 'full_name', 'employee_id', 'id', 'title', 'role']
        section_title = "Data Record"
        
        for field in name_fields:
            if field in row_data:
                section_title = f"{field.replace('_', ' ').title()}: {row_data[field]}"
                break
        
        chunks.append({
            "content": content,
            "section_title": section_title,
            "heading_level": 1,
            "chunk_index": index,
            "total_chunks": len(df),
            "row_data": row_data  # Keep structured data for potential use
        })
    
    return chunks


def load_document(file_path: str, **kwargs) -> List[Dict[str, Any]]:
    """Universal document loader that handles different file types."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.md':
        return load_markdown(str(file_path), **kwargs)
    elif file_path.suffix.lower() == '.csv':
        return load_csv(str(file_path), **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
