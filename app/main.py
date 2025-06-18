from typing import Dict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.services.vector_store import (
    search_documents, 
    init_qdrant, 
    index_documents, 
    get_collection_stats,
    delete_collection
)
from app.services.rag_engine import generate_response, generate_structured_response
from app.schemas.user import UserCreate
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.db.models import User


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # frontend url 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBasic()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# # Dummy user database
# users_db: Dict[str, Dict[str, str]] = {
#     "Tony": {"password": "password123", "role": "engineering"},
#     "Bruce": {"password": "securepass", "role": "marketing"},
#     "Sam": {"password": "financepass", "role": "finance"},
#     "Peter": {"password": "pete123", "role": "engineering"},
#     "Sid": {"password": "sidpass123", "role": "marketing"},
#     "Natasha": {"passwoed": "hrpass123", "role": "hr"},
#      "Elena": {"password": "execpass", "role": "c-level"},
# }


# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == credentials.username).first()
    if not user or user.password != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": user.username, "role": user.role}


# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}


# Protected chat endpoint
@app.post("/chat")
def query(user=Depends(authenticate), message: str = "Hello"):
    role = user["role"]
    docs = search_documents(message, role)
    if not docs:
        return {"response": "Sorry, no relevant documents were found."}
    # Use enhanced RAG engine
    result = generate_structured_response(message, docs, user_role=role)
    return result



def require_role(allowed_roles: list[str]):
    def role_checker(user=Depends(authenticate)):
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="Not authorized")
        return user
    return role_checker



@app.get("/admin/users")
def list_users(
    db: Session = Depends(get_db),
    user=Depends(require_role(["admin"]))
):
    users = db.query(User).all()
    return [{"username": u.username, "role": u.role} for u in users]


@app.post("/admin/users")
def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    user=Depends(require_role(["admin"]))
):
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = User(**user_data.dict())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User added", "username": new_user.username, "role": new_user.role}


@app.delete("/admin/users/{username}")
def delete_user(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(require_role(["admin"]))
):
    u = db.query(User).filter(User.username == username).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(u)
    db.commit()
    return {"message": f"User '{username}' deleted"}


# =============================================================================
# INDEXING ENDPOINTS
# =============================================================================

@app.post("/admin/index/init")
def initialize_vector_store(user=Depends(require_role(["admin", "c-level"]))):
    """Initialize the Qdrant vector store collection."""
    try:
        init_qdrant()
        return {"message": "Vector store initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize vector store: {str(e)}")


@app.post("/admin/index/documents")
def index_all_documents(
    batch_size: int = 50,
    user=Depends(require_role(["admin", "c-level"]))
):
    """Index all documents from the resources/data directory."""
    try:
        total_chunks = index_documents(batch_size=batch_size)
        return {
            "message": "Document indexing completed successfully",
            "total_chunks": total_chunks,
            "batch_size": batch_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {str(e)}")


@app.get("/admin/index/stats")
def get_indexing_stats(user=Depends(require_role(["admin", "c-level"]))):
    """Get statistics about the indexed documents."""
    try:
        stats = get_collection_stats()
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/admin/index/clear")
def clear_vector_store(user=Depends(require_role(["admin", "c-level"]))):
    """Clear the entire vector store collection (use with caution)."""
    try:
        success = delete_collection()
        if success:
            return {"message": "Vector store cleared successfully"}
        else:
            return {"message": "Vector store was already empty or does not exist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {str(e)}")


@app.post("/admin/index/reindex")
def reindex_all_documents(
    batch_size: int = 50,
    user=Depends(require_role(["admin", "c-level"]))
):
    """Clear and reindex all documents."""
    try:
        # Clear existing collection
        delete_collection()
        
        # Initialize new collection
        init_qdrant()
        
        # Index documents
        total_chunks = index_documents(batch_size=batch_size)
        
        return {
            "message": "Document reindexing completed successfully",
            "total_chunks": total_chunks,
            "batch_size": batch_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reindex documents: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)