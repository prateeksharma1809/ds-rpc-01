import os
import glob
from pathlib import Path
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, VectorParams, Distance, PointStruct, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


from app.utils.file_loader import load_markdown, load_csv
from app.services.rag_engine import generate_response

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_rbac_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# Initialize model and client
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_qdrant():
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def index_documents(data_dir="resources/data"):
    """Walk through data folders and index docs based on role."""
    documents = []
    for role_dir in Path(data_dir).iterdir():
        if role_dir.is_dir():
            role = role_dir.name
            for file_path in glob.glob(f"{role_dir}/*"):
                print("Processing :", file_path)
                if file_path.endswith(".md"):
                    texts = load_markdown(file_path)
                elif file_path.endswith(".csv"):
                    texts = load_csv(file_path)
                else:
                    print('file cannot be processed', file_path)
                    continue
                for chunk in texts:
                    vector = embedding_model.encode(chunk["content"]).tolist()
                    documents.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "role": role,
                                "source": Path(file_path).name,
                                "section_title": chunk["section_title"],
                                "heading_level": chunk["heading_level"],
                                "content": chunk["content"], 
                            }
                        )
                    )
    client.upload_points(collection_name=COLLECTION_NAME, points=documents)
    print(f"Indexed {len(documents)} chunks into Qdrant.")


def search_documents(query: str, role: str, k: int = 3):
    query_vector = embedding_model.encode(query).tolist()

    if role == "c-level":
        search_filter = None
    else:
        search_filter = Filter(must=[],
            should=[
                FieldCondition(key="role", match=MatchValue(value=role)),
                FieldCondition(key="role", match=MatchValue(value="general"))
            ])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        query_filter=search_filter,
        with_payload=True
    ).points

    return [
        {
            "score": round(r.score, 3),
            "content": r.payload.get("content", ""),
            "source": r.payload.get("source", ""),
            "section_title": r.payload.get("section_title", ""),
            "heading_level": r.payload.get("heading_level", ""),
        }
        for r in results
    ]

if __name__ == "__main__":
    # client.delete_collection(collection_name=COLLECTION_NAME)
    init_qdrant()
    index_documents()
    # query = "Show me Q4 2024 financial results"
    # role = "finance"
    # docs = search_documents(query, role,3)

    # print(f"\nTop Matching Documents:")
    # if not docs:
    #     print("No documents found.")
    # else:
    #     for doc in docs:
    #         print(f"[{doc['score']}] Title: {doc['section_title']} - Level {doc['heading_level']}")
    #         print(doc["content"][:300], "\n")
    # print("\nFinal Response:")
    # answer = generate_response(query, docs)
    # print(answer)