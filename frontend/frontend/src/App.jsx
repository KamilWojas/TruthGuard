import { useState } from "react";
import axios from "axios";

function App() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeText = async () => {
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze_text", {
        text: input,
      });
      setResult(response.data);
    } catch (error) {
      console.error("Błąd analizy:", error);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold text-gray-800">TruthGuard - Fake News Detector</h1>

      <textarea
        className="mt-4 p-3 w-full max-w-2xl border rounded-lg"
        placeholder="Wpisz tekst do analizy..."
        rows="4"
        value={input}
        onChange={(e) => setInput(e.target.value)}
      ></textarea>

      <button
        onClick={analyzeText}
        className="mt-4 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition"
        disabled={loading}
      >
        {loading ? "Analizowanie..." : "Sprawdź Fake News"}
      </button>

      {result && (
        <div className="mt-6 p-4 w-full max-w-2xl bg-white shadow-md rounded-lg">
          <h2 className="text-xl font-bold">Wynik analizy:</h2>
          <p><strong>Tekst:</strong> {result.text}</p>
          <p><strong>Prawdopodobieństwo Fake News:</strong> {result.fake_news_score.toFixed(2)}</p>
          <p><strong>Wiarygodność źródła:</strong> {result.source_reliability.toFixed(2)}</p>
          <p><strong>Klasyfikacja:</strong> {result.classification}</p>
        </div>
      )}
    </div>
  );
}
export default App;

