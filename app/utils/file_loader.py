import markdown
import re
import pandas as pd

def load_markdown(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Case 1: No headers or separators — treat whole file as one chunk
    if not re.search(r'^#{2,3} |^-{3,}$', text, flags=re.MULTILINE):
        return [{
            "content": text,
            "section_title": "Full Document",
            "heading_level": 0
        }]

    # Case 2: Use `##`, `###` headers or `----` separators
    chunks = []

    # Split on `##`, `###`, or `----` as section markers
    raw_chunks = re.split(r'(?=^#{2,3} .*|^-{3,}$)', text, flags=re.MULTILINE)

    current_title = None
    heading_level = 1

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk.split()) < 20:
            continue

        lines = chunk.split("\n")
        first_line = lines[0].strip()

        # Case: markdown heading
        match = re.match(r'^(#{2,3}) (.+)', first_line)
        if match:
            hashes, section_title = match.groups()
            heading_level = len(hashes)
            body = "\n".join(lines[1:]).strip()
        elif re.match(r'^-{3,}$', first_line):  # separator
            section_title = "Section"
            heading_level = 2
            body = "\n".join(lines[1:]).strip()
        else:
            # continuation of previous section or generic
            section_title = current_title or "Unnamed Section"
            body = chunk

        if len(body.split()) < 20:
            continue

        current_title = section_title
        chunks.append({
            "content": body,
            "section_title": section_title,
            "heading_level": heading_level
        })

    return chunks

def load_csv(file_path: str) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(file_path)
    rows = df.astype(str).apply(lambda x: ", ".join(x), axis=1).tolist()
    return [{
        "content": row,
        "section_title": "CSV Row",
        "heading_level": 1
    } for row in rows if len(row.split()) > 16]