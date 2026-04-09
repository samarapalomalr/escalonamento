from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Se quiser mudar para Postgres depois, é só trocar esta string
# Exemplo Postgres: "postgresql://usuario:senha@localhost/dbname"
SQLALCHEMY_DATABASE_URL = "sqlite:///./patrimonio.db"

# O check_same_thread é necessário apenas para o SQLite.
# Para outros bancos, esse argumento pode ser removido.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Cria uma fábrica de sessões para o banco de dados
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Classe base para a criação dos modelos (mapeamento ORM)
Base = declarative_base()

# Dependência para as rotas do FastAPI. 
# Garante que a conexão feche após a requisição.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()