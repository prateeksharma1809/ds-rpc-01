from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    password: str
    role: str