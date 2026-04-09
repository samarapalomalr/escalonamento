import os
import sys  
import uvicorn

# Adiciona o diretório atual ao PATH antes de importar a pasta 'app'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List

# --- AGORA OS IMPORTS DA ESTRUTURA backend/app/ ---
from app import models, schemas, database, auth
from app.services import ai_service

# Inicializa o banco de dados
print("Iniciando banco de dados...")
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Patrimônio IA API")

# CONFIGURAÇÃO DE CORS - CRITICAL PARA VERCEL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AJUSTE NO BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(BASE_DIR, "assets")

if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
else:
    print(f"Aviso: Pasta assets não encontrada em {assets_path}")

# --- ROTAS ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "Patrimônio IA API rodando com sucesso"}

@app.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.Usuario).filter(models.Usuario.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Usuário já cadastrado.")
    
    hashed_password = auth.get_password_hash(user.password)
    novo_usuario = models.Usuario(username=user.username, password_hash=hashed_password)
    
    db.add(novo_usuario)
    db.commit()
    db.refresh(novo_usuario)
    return novo_usuario

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.Usuario).filter(models.Usuario.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }

@app.post("/classify", response_model=schemas.DeteccaoResponse)
async def classify_and_save(
    categoria: str = Form(...),
    user_id: int = Form(...),
    latitude: float = Form(None),
    longitude: float = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db)
):
    try:
        image_bytes = await file.read()
        label, confidence = ai_service.predict_image(image_bytes, categoria)

        nova_deteccao = models.Deteccao(
            elemento=label,
            categoria=categoria,
            data=datetime.utcnow(),
            latitude=latitude,
            longitude=longitude,
            confianca_modelo=confidence,
            user_id=user_id
        )

        db.add(nova_deteccao)
        db.commit()
        db.refresh(nova_deteccao)
        return nova_deteccao
    except Exception as e:
        print(f"ERRO NA CLASSIFICAÇÃO: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}", response_model=List[schemas.DeteccaoResponse])
def get_history(user_id: int, db: Session = Depends(database.get_db)):
    return db.query(models.Deteccao).filter(models.Deteccao.user_id == user_id).order_by(models.Deteccao.data.desc()).all()

# EXECUÇÃO DO SERVIDOR
if __name__ == "__main__":
    # O Render exige que a porta seja lida da variável de ambiente PORT
    # e que o host seja 0.0.0.0
    port = int(os.environ.get("PORT", 10000))
    print(f"Servidor subindo na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)