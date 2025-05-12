import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

# Load FAISS index and metadata
try:
    index = faiss.read_index("bhagavatam_faiss.index")
    df = pickle.load(open("bhagavatam_metadata.pkl", "rb"))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load AI Summarization Model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Load AI Explanation Model
    explanation_model = pipeline("text2text-generation", model="google/flan-t5-large")

    st.title("Bhagavatam AI Assistant 📖")
    st.write("Enter a verse, and AI will provide a simplified explanation.")

    query = st.text_input("🔍 Enter your verse:")

    if query:
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, 1)  # Get the closest verse

        st.subheader("Verse Found:")
        idx = indices[0][0]
        if idx < len(df):  # Prevent index errors
            verse = df.iloc[idx]

            st.write(f"### 📖 Canto {verse['Canto']}, Chapter {verse['Chapter']}")
            st.write(f"🔸 **Sanskrit:** {verse['Sanskrit']}")
            st.write(f"🔹 **Translation:** {verse['Translation']}")

            # AI Simplified Explanation
            explanation_prompt = f"Explain this Bhagavatam verse in simple words: {verse['Translation']}"
            explanation = explanation_model(explanation_prompt, max_length=100, do_sample=False)
            st.write(f"🧠 **AI Explanation:** {explanation[0]['generated_text']}")

            # Display Purport and AI-generated Summary
            if "Purport" in verse and isinstance(verse["Purport"], str):
                purport_text = verse["Purport"]
                st.write(f"📜 **Purport:** {purport_text}")

                purport_summary = summarizer(purport_text, max_length=100, min_length=30, do_sample=False)
                st.write(f"🔍 **AI Summary (Purport):** {purport_summary[0]['summary_text']}")

except Exception as e:
    st.error(f"Error loading data: {e}")

#streamlit run app.py
