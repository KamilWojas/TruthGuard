from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Wczytanie modelu NLP do analizy fake newsów
nlp_model = pipeline("text-classification", model="facebook/bart-large-mnli")


# Model wejściowy dla analizy tekstu
class TextAnalysisRequest(BaseModel):
    text: str


# Model odpowiedzi API
class AnalysisResponse(BaseModel):
    text: str
    fake_news_score: float
    source_reliability: float
    classification: str


@app.post("/analyze_text", response_model=AnalysisResponse)
def analyze_text(request: TextAnalysisRequest):
    """
    Endpoint do analizy tekstu pod kątem fake newsów.
    Zwraca ocenę prawdopodobieństwa fałszywej informacji oraz ocenę wiarygodności źródła.
    """
    # Analiza NLP
    result = nlp_model(request.text)
    classification_label = result[0]['label']
    fake_news_score = result[0]['score'] if classification_label == "FAKE" else (1 - result[0]['score'])
    source_reliability = 1 - fake_news_score

    return AnalysisResponse(
        text=request.text,
        fake_news_score=fake_news_score,
        source_reliability=source_reliability,
        classification=classification_label
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
