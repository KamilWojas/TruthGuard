from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Wczytanie modelu NLP do analizy fake newsów
nlp_model = pipeline("text-classification", model="facebook/bart-large-mnli")

# Model wejściowy dla anlizy tekstów
class TextAnalystRequest(BaseModel):
    text: str

# Model odpowiedzi API
class AnalysisResponse(BaseModel):
    text: str
    fake_news_score: float
    source_reliability: float
    classification: str

@app.post("/analyst_text", response_model=AnalysisResponse)
def analyze_text(request: TextAnalystRequest):
    """
     Endpoint do analizy tekstu pod kątem fake newsów.
    Zwraca ocenę prawdopodobieństwa fałszywej informacji oraz ocenę wiarygodności źródła.
        """