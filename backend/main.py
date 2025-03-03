from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from transformers import pipeline
from backend.database import SessionLocal, init_db, AnalyzedText

app = FastAPI()

# 🔹 Wczytanie modelu NLP do analizy fake newsów
nlp_model = pipeline("text-classification", model="mrm8488/bert-mini-fakenews")

# 🔹 Inicjalizacja bazy danych
init_db()


# 🔹 Funkcja do pobierania sesji bazy danych
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 🔹 Model wejściowy dla analizy tekstu
from pydantic import BaseModel


class TextAnalysisRequest(BaseModel):
    text: str


class AnalysisResponse(BaseModel):
    text: str
    fake_news_score: float
    source_reliability: float
    classification: str

@app.get("/")
def home():
    return {"message": "TruthGuard API działa! Sprawdź /docs"}

@app.post("/analyze_text", response_model=AnalysisResponse)
def analyze_text(request: TextAnalysisRequest, db: Session = Depends(get_db)):
    """
    Endpoint do analizy tekstu pod kątem fake newsów.
    Zwraca ocenę prawdopodobieństwa fałszywej informacji oraz ocenę wiarygodności źródła.
    """
    # Analiza NLP
    result = nlp_model(request.text)
    classification_label = result[0]['label']
    fake_news_score = result[0]['score'] if classification_label == "FAKE" else (1 - result[0]['score'])
    source_reliability = 1 - fake_news_score

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
        classification=classification_label
    )
