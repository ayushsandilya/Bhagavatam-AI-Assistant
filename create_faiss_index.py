import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load dataset
df = pd.read_csv("bhagavatam_data.csv", encoding="utf-8-sig")

# Identify correct columns (handling variations in column names)
print("Columns in dataset:", df.columns.tolist())  # Debugging: Print available columns

# Expected columns
expected_cols = ["Canto", "Chapter", "Sanskrit", "Translation", "Purport"]
actual_cols = {col.lower(): col for col in df.columns}

# Ensure all required columns exist
missing_cols = [col for col in expected_cols if col.lower() not in actual_cols]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Use actual column names (handling case differences)
canto_col = actual_cols["canto"]
chapter_col = actual_cols["chapter"]
sanskrit_col = actual_cols["sanskrit"]
translation_col = actual_cols["translation"]
purport_col = actual_cols["purport"]

# Keep only required columns and drop missing values
df = df[[canto_col, chapter_col, sanskrit_col, translation_col, purport_col]].dropna()

# Load sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode all Sanskrit verses
embeddings = model.encode(df[sanskrit_col].tolist(), convert_to_numpy=True)

# Create FAISS index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(embeddings)  # Add embeddings to FAISS index

# Save FAISS index and metadata (including Purport)
faiss.write_index(index, "bhagavatam_faiss.index")
with open("bhagavatam_metadata.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ FAISS index created and saved successfully with Purport!")
