from sqlalchemy import create_engine, text, Column, Integer, BigInteger, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import JSONB   # ✅ add this
from pgvector.sqlalchemy import Vector
from .deps import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# NOTE: Vector(dim) must match settings.EMBED_DIM
class Document(Base):
    __tablename__ = "documents"
    id = Column(BigInteger, primary_key=True)
    doc_id = Column(String, nullable=False)
    chunk_id = Column(Integer, nullable=False)
    content = Column(String, nullable=False)
    # ✅ map to JSONB but keep DB column name "metadata"
    meta = Column("metadata", JSONB, nullable=True)
    embedding = Column(Vector(settings.EMBED_DIM))

def init_db():
    # Ensure pgvector exists, then create tables if missing
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
