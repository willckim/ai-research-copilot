from sqlalchemy import create_engine, text, Column, Integer, BigInteger, String
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from .deps import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# NOTE: Vector(dim) must match your embedding dimension (settings.EMBED_DIM)
class Document(Base):
    __tablename__ = "documents"
    id = Column(BigInteger, primary_key=True)
    doc_id = Column(String, nullable=False)
    chunk_id = Column(Integer, nullable=False)
    content = Column(String, nullable=False)
    # ⚠️ Don't use reserved name "metadata" in Python class
    meta = Column("metadata", String, nullable=True)  # maps to DB column "metadata"
    embedding = Column(Vector(settings.EMBED_DIM))

def init_db():
    Base.metadata.create_all(engine)
