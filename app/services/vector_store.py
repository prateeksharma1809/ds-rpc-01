import os
import glob
from pathlib import Path
import uuid
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, VectorParams, Distance, PointStruct, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from app.utils.file_loader import load_document, load_markdown, load_csv
from app.services.rag_engine import generate_response

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_rbac_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Initialize model and client
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_qdrant():
    """Initialize Qdrant collection with proper configuration."""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        vector_size = embedding_model.get_sentence_embedding_dimension()
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size, 
                distance=Distance.COSINE,
                on_disk=True  # Store vectors on disk for large collections
            )
        )
        logger.info(f"Created collection: {COLLECTION_NAME} with vector size: {vector_size}")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists.")

def index_documents(data_dir="resources/data", batch_size=100):
    """Walk through data folders and index docs based on role with improved processing."""
    documents = []
    total_files = 0
    total_chunks = 0
    
    for role_dir in Path(data_dir).iterdir():
        if role_dir.is_dir():
            role = role_dir.name
            logger.info(f"Processing role directory: {role}")
            
            for file_path in glob.glob(f"{role_dir}/*"):
                file_path = Path(file_path)
                total_files += 1
                
                logger.info(f"Processing file: {file_path}")
                
                try:
                    # Use the improved document loader
                    if file_path.suffix.lower() == '.md':
                        texts = load_markdown(str(file_path), max_chunk_size=400, overlap=50)
                    elif file_path.suffix.lower() == '.csv':
                        texts = load_csv(str(file_path), max_chunk_size=300)
                    else:
                        logger.warning(f'Unsupported file type: {file_path}')
                        continue
                    
                    logger.info(f"Extracted {len(texts)} chunks from {file_path.name}")
                    
                    for chunk in texts:
                        # Skip very short chunks
                        if len(chunk["content"].split()) < 10:
                            continue
                        
                        # Create vector embedding
                        vector = embedding_model.encode(chunk["content"]).tolist()
                        
                        # Create enhanced payload with metadata
                        payload = {
                            "role": role,
                            "source": file_path.name,
                            "section_title": chunk["section_title"],
                            "heading_level": chunk["heading_level"],
                            "content": chunk["content"],
                            "chunk_index": chunk.get("chunk_index", 0),
                            "total_chunks": chunk.get("total_chunks", -1),
                            "file_type": file_path.suffix.lower(),
                            "word_count": len(chunk["content"].split()),
                            "role_directory": role
                        }
                        
                        # Add row_data for CSV files if available
                        if "row_data" in chunk:
                            payload["row_data"] = chunk["row_data"]
                        
                        documents.append(
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector=vector,
                                payload=payload
                            )
                        )
                        
                        total_chunks += 1
                        
                        # Upload in batches to avoid memory issues
                        if len(documents) >= batch_size:
                            client.upload_points(collection_name=COLLECTION_NAME, points=documents)
                            logger.info(f"Uploaded batch of {len(documents)} documents")
                            documents = []
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
    
    # Upload remaining documents
    if documents:
        client.upload_points(collection_name=COLLECTION_NAME, points=documents)
        logger.info(f"Uploaded final batch of {len(documents)} documents")
    
    logger.info(f"Indexing complete. Processed {total_files} files, created {total_chunks} chunks.")
    return total_chunks


def search_documents(query: str, role: str, k: int = 5, min_score: float = 0.3):
    """Search documents with improved filtering and scoring."""
    query_vector = embedding_model.encode(query).tolist()

    # Create search filter based on role
    if role == "c-level":
        # C-level can access all documents
        search_filter = None
    else:
        # Other roles can access their role-specific documents and general documents
        search_filter = Filter(
            should=[
                FieldCondition(key="role", match=MatchValue(value=role)),
                FieldCondition(key="role", match=MatchValue(value="general"))
            ]
        )

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k * 2,  # Get more results to filter by score
            query_filter=search_filter,
            with_payload=True,
            score_threshold=min_score
        ).points

        # Filter and format results
        filtered_results = []
        for r in results:
            if r.score >= min_score:
                filtered_results.append({
                    "score": round(r.score, 3),
                    "content": r.payload.get("content", ""),
                    "source": r.payload.get("source", ""),
                    "section_title": r.payload.get("section_title", ""),
                    "heading_level": r.payload.get("heading_level", ""),
                    "role": r.payload.get("role", ""),
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "total_chunks": r.payload.get("total_chunks", -1),
                    "word_count": r.payload.get("word_count", 0),
                    "file_type": r.payload.get("file_type", ""),
                    "row_data": r.payload.get("row_data", {})
                })
        
        # Return top k results
        return filtered_results[:k]
    
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return []


def get_collection_stats():
    """Get detailed statistics about the collection."""
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        
        # Get some sample points to analyze content
        # Use a simple query to get sample points
        sample_points = client.query_points(
            collection_name=COLLECTION_NAME,
            query=[0] * embedding_model.get_sentence_embedding_dimension(),
            limit=100,
            with_payload=True,
            score_threshold=0.0  # Get all points regardless of score
        ).points
        
        # Analyze role distribution
        role_counts = {}
        file_type_counts = {}
        total_words = 0
        
        for point in sample_points:
            role = point.payload.get("role", "unknown")
            file_type = point.payload.get("file_type", "unknown")
            word_count = point.payload.get("word_count", 0)
            
            role_counts[role] = role_counts.get(role, 0) + 1
            file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
            total_words += word_count
        
        return {
            "total_points": collection_info.points_count,
            "collection_name": COLLECTION_NAME,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": collection_info.config.params.vectors.distance.value,
            "role_distribution": role_counts,
            "file_type_distribution": file_type_counts,
            "avg_words_per_chunk": total_words / len(sample_points) if sample_points else 0,
            "sample_size": len(sample_points)
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        return {"error": str(e)}


def delete_collection():
    """Delete the collection."""
    try:
        if client.collection_exists(collection_name=COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Deleted collection: {COLLECTION_NAME}")
            return True
        else:
            logger.info(f"Collection {COLLECTION_NAME} does not exist")
            return False
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        return False


if __name__ == "__main__":
    # Initialize and index documents
    init_qdrant()
    total_chunks = index_documents()
    
    # Get and display statistics
    stats = get_collection_stats()
    print(f"\nCollection Statistics:")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Collection info: {stats}")
    
    # Test search
    test_queries = [
        "microservices architecture",
        "employee performance data",
        "marketing campaign results",
        "financial analysis"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        results = search_documents(query, "engineering", k=3)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['score']}] {result['section_title']} ({result['role']})")
            print(f"     Content: {result['content'][:100]}...")