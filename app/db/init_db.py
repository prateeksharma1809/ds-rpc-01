from app.db.models import Base, User
from app.db.session import engine, SessionLocal

# Create tables
Base.metadata.create_all(bind=engine)

# Seed users
users_to_seed = [
    {"username": "Tony", "password": "password123", "role": "engineering"},
    {"username": "Bruce", "password": "securepass", "role": "marketing"},
    {"username": "Sam", "password": "financepass", "role": "finance"},
    {"username": "Peter", "password": "pete123", "role": "engineering"},
    {"username": "Sid", "password": "sidpass123", "role": "marketing"},
    {"username": "Natasha", "password": "hrpass123", "role": "hr"},
    {"username": "Elena", "password": "execpass", "role": "c-level"},
    {"username": "Nina", "password": "employee123", "role": "general"} ,
    {"username":"admin", "password":"adminpass", "role":"admin"}
]

db = SessionLocal()

for user_data in users_to_seed:
    existing = db.query(User).filter(User.username == user_data["username"]).first()
    if not existing:
        db.add(User(**user_data))

db.commit()
db.close()
print(" Seeded users successfully.")
