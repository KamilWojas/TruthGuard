from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from transformers import pipeline
from pydantic import BaseModel
from newspaper import Article  # 📌 Nowa biblioteka do pobierania treści z URL
from backend.database import SessionLocal, init_db, AnalyzedText

app = FastAPI()

# 🔹 Wczytanie modelu NLP do analizy fake newsów
nlp_model = pipeline("text-classification", model="facebook/bart-large-mnli")

# 🔹 Inicjalizacja bazy danych
init_db()

# 🔹 Funkcja do pobierania sesji bazy danych
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 🔹 Model wejściowy dla analizy tekstu i URL
class TextAnalysisRequest(BaseModel):
    text: str = None  # 📌 Opcjonalnie użytkownik może podać tekst...
    url: str = None   # 📌 ...lub URL do analizy

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
    Obsługuje zarówno tekst jak i URL artykułu.
    """

    if request.url:  # 📌 Jeśli użytkownik podał URL → pobieramy treść strony
        try:
            article = Article(request.url)
            article.download()
            article.parse()
            request.text = article.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Błąd pobierania treści z URL: {str(e)}")

    if not request.text:
        raise HTTPException(status_code=400, detail="Brak tekstu do analizy!")

    # 🔹 Analiza NLP
    result = nlp_model(request.text)
    classification_label = result[0]['label']
    fake_news_score = result[0]['score'] if classification_label == "FAKE" else (1 - result[0]['score'])
    source_reliability = 1 - fake_news_score

    # 🔹 Zapisanie wyniku do bazy danych
    analyzed_text = AnalyzedText(
        text=request.text[:500],  # 📌 Ograniczamy długość zapisanego tekstu
        fake_news_score=fake_news_score,
        source_reliability=source_reliability,
        classification=classification_label
    )
    db.add(analyzed_text)
    db.commit()

    return AnalysisResponse(
        text=request.text[:500],  # 📌 Ograniczamy długość zwracanego tekstu
        fake_news_score=fake_news_score,
        source_reliability=source_reliability,
        classification=classification_label
    )
