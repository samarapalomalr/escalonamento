from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Usuario(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    
    # Relacionamento com as detecções
    deteccoes = relationship("Deteccao", back_populates="proprietario")

class Deteccao(Base):
    __tablename__ = "deteccoes"

    id = Column(Integer, primary_key=True, index=True)
    elemento = Column(String)
    categoria = Column(String)
    data = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    confianca_modelo = Column(Float, nullable=True)
    
    user_id = Column(Integer, ForeignKey("users.id"))
    proprietario = relationship("Usuario", back_populates="deteccoes")