from typing import Optional
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from transformers import pipeline
from backend.database import SessionLocal, init_db, AnalyzedText
import json
import tldextract
from pydantic import BaseModel

app = FastAPI()

# 🔹 Wczytanie modelu NLP do analizy fake newsów
nlp_model = pipeline("text-classification", model="mrm8488/bert-mini-fakenews")

# 🔹 Inicjalizacja bazy danych
init_db()

# 🔹 Wczytujemy bazę źródeł fake newsów
with open("backend/sources.json", "r") as f:
    sources_data = json.load(f)

trusted_sources = sources_data["trusted_sources"]
fake_news_sources = sources_data["fake_news_sources"]


def get_db():
    """ Pobieranie sesji bazy danych """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 🔹 Model wejściowy dla analizy tekstu
class TextAnalysisRequest(BaseModel):
    text: str
    url: Optional[str] = None  # Opcjonalny adres URL źródła


class AnalysisResponse(BaseModel):
    text: str
    fake_news_score: float
    source_reliability: float
    classification: str
    source_status: str


def check_source_reliability(url: str) -> str:
    """
    Sprawdza, czy źródło informacji pochodzi z listy wiarygodnych lub podejrzanych stron.
    """
    domain = tldextract.extract(url).registered_domain

    if domain in trusted_sources:
        return "trusted"
    elif domain in fake_news_sources:
        return "fake"
    else:
        return "unknown"


@app.get("/")
def home():
    return {"message": "TruthGuard API działa! Sprawdź /docs"}


@app.post("/analyze_text", response_model=AnalysisResponse)
def analyze_text(request: TextAnalysisRequest, db: Session = Depends(get_db)):
    """
    Endpoint do analizy tekstu pod kątem fake newsów oraz oceny źródła.
    """
    # 🔹 Analiza NLP
    result = nlp_model(request.text)
    classification_label = result[0]['label']
    fake_news_score = result[0]['score'] if classification_label == "FAKE" else (1 - result[0]['score'])
    source_reliability = 1 - fake_news_score

    # 🔹 Sprawdzenie wiarygodności źródła (jeśli podano URL)
    source_status = "unknown"
    if request.url:
        source_status = check_source_reliability(request.url)

    # 🔹 Zapisanie wyniku do bazy danych
    analyzed_text = AnalyzedText(
        text=request.text,
        fake_news_score=fake_news_score,
        source_reliability=source_reliability,
        classification=classification_label
    )
    db.add(analyzed_text)
    db.commit()

    return AnalysisResponse(
        text=request.text,
        fake_news_score=fake_news_score,
        source_reliability=source_reliability,
        classification=classification_label,
        source_status=source_status
    )
