from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List

# --- USUÁRIO ---

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    model_config = ConfigDict(from_attributes=True)

# MODIFICAÇÃO: Esquema de Token expandido para incluir dados do usuário no login
class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str

# --- DETECÇÃO ---

class DeteccaoBase(BaseModel):
    elemento: str
    categoria: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confianca_modelo: Optional[float] = None

class DeteccaoResponse(DeteccaoBase):
    id: int
    data: datetime
    user_id: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)