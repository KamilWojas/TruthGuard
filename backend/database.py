from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# 🔹 Dane do połączenia z PostgreSQL
DATABASE_URL = "postgresql://truthguard_user:super_secure_password@localhost/truthguard_db"

# 🔹 Tworzenie silnika bazy danych
engine = create_engine(DATABASE_URL)

# 🔹 Tworzenie sesji
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 🔹 Definiowanie bazy SQLAlchemy
Base = declarative_base()

# 🔹 Model bazy danych dla analizy fake newsów
class AnalyzedText(Base):
    __tablename__ = "analyzed_texts"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    fake_news_score = Column(Float, nullable=False)
    source_reliability = Column(Float, nullable=False)
    classification = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# 🔹 Tworzenie tabeli w bazie danych
def init_db():
    Base.metadata.create_all(bind=engine)
