import pandas as pd
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def load_faiss_index():
    """Load FAISS index and metadata."""
    try:
        index = faiss.read_index("bhagavatam_faiss.index")
        df = pickle.load(open("bhagavatam_metadata.pkl", "rb"))
        return index, df
    except FileNotFoundError:
        print("❌ Error: FAISS index or metadata file not found! Please run 'create_faiss_index.py' first.")
        return None, None

# Load AI Models
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
explanation_model = pipeline("text2text-generation", model="google/flan-t5-large")

def search_verse(query):
    """Search and return relevant Bhagavatam verse with AI explanations and summaries."""
    index, df = load_faiss_index()
    if index is None or df is None:
        return None
    
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 1)  # Get the closest verse
    idx = indices[0][0]
    
    if idx >= len(df):
        print("❌ No matching verse found.")
        return None
    
    verse = df.iloc[idx]
    
    # AI Simplified Explanation
    explanation_prompt = f"Explain this Bhagavatam verse in simple words: {verse['Translation']}"
    explanation = explanation_model(explanation_prompt, do_sample=False)[0]['generated_text']
    
    # AI Summary for Purport
    purport_text = verse.get("Purport", "No purport available.")
    if isinstance(purport_text, str) and purport_text.strip():
        purport_summary = summarizer(purport_text, do_sample=False)[0]['summary_text']
    else:
        purport_summary = "No summary available."
    
    return {
        "Canto": verse["Canto"],
        "Chapter": verse["Chapter"],
        "Sanskrit": verse["Sanskrit"],
        "Translation": verse["Translation"],
        "Purport": purport_text,
        "AI Explanation": explanation,
        "AI Summary": purport_summary,
        "Similarity Score": distances[0][0]
    }

if __name__ == "__main__":
    query = input("🔍 Enter your query: ")
    result = search_verse(query)
    
    if result:
        print(f"\n📖 **Canto {result['Canto']}, Chapter {result['Chapter']}**")
        print(f"🔸 **Sanskrit:** {result['Sanskrit']}")
        print(f"🔹 **Translation:** {result['Translation']}")
        print(f"📜 **Purport:** {result['Purport']}")
        print(f"🧠 **AI Explanation:** {result['AI Explanation']}")
        print(f"🔍 **AI Summary (Purport):** {result['AI Summary']}")
        print(f"⭐ **Similarity Score:** {result['Similarity Score']:.4f}")