import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="🛒 E-commerce Product Recommender", layout="centered")
st.title("🛍️ E-commerce Product Recommender System")
st.markdown("Enter a product description and get similar items based on their descriptions.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Dheenu\\data.csv", encoding='ISO-8859-1')
    df.rename(columns={'Description': 'description'}, inplace=True)
    df = df.dropna(subset=['description']).drop_duplicates(subset=['description']).reset_index(drop=True)
    if 'title' not in df.columns:
        df['title'] = df['StockCode'].astype(str)
    return df

data = load_data()

# Load model and compute embeddings
@st.cache_resource
def load_model_and_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['description'].tolist(), show_progress_bar=True)
    return model, embeddings

model, embeddings = load_model_and_embeddings(data)

# Search function
def get_similar_products(query, top_n=10):
    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = sims.argsort()[-top_n:][::-1]
    recommendations = data.iloc[top_indices][['title', 'description']].copy()
    recommendations['similarity'] = sims[top_indices]
    return recommendations

# Streamlit input + output
with st.form("recommender_form"):
    user_input = st.text_input("🔍 Enter product description", "")
    submit = st.form_submit_button("Recommend")

if submit:
    if not user_input.strip():
        st.warning("Please enter a product description.")

    else:
        results = get_similar_products(user_input)
        st.subheader("📦 Recommended Products")
        st.dataframe(results, use_container_width=True)
