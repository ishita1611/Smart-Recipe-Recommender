import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv('final_dataset.csv')
    df['Ingredients'] = df['Ingredients'].fillna('')
    df['cleaned_ingredients'] = df['Ingredients'].apply(
        lambda txt: ' '.join(i.strip().lower() for i in re.split(',|;|\n', str(txt)) if i.strip())
    )
    return df

df = load_data()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_ingredients'])

# DBSCAN Clustering – loosened up
dbscan = DBSCAN(eps=0.6, min_samples=2, metric='cosine')
df['cluster'] = dbscan.fit_predict(X)

# Debug info
st.sidebar.title("🛠️ Debug Info")
st.sidebar.write("Clusters found:", df['cluster'].nunique())
st.sidebar.write("Noise points (-1):", (df['cluster'] == -1).sum())

# Recipe Recommender
def recommend_recipes(user_input, top_n=5):
    cleaned_input = ' '.join(i.strip().lower() for i in re.split(',|;|\n', user_input) if i.strip())
    input_vec = vectorizer.transform([cleaned_input])
    
    distances = cosine_distances(input_vec, X)
    closest_index = distances.argmin()
    predicted_cluster = df.iloc[closest_index]['cluster']

    if predicted_cluster == -1:
        return pd.DataFrame()  # no cluster match

    cluster_recipes = df[df['cluster'] == predicted_cluster]
    return cluster_recipes[['Title', 'Ingredients']].head(top_n)

# Streamlit UI
st.title("🍝 AI Recipe Recommender")
st.markdown("Find the best recipes based on the ingredients you have!")

user_input = st.text_input("📝 Enter your ingredients (comma-separated):")

if user_input:
    results = recommend_recipes(user_input)

    if results.empty:
        st.warning("😔 Sorry, your ingredients don't match any recipe cluster.")
    else:
        st.success("🎉 Found matching recipes:")
        for _, row in results.iterrows():
            st.markdown(f"**🍽️ {row['Title']}**")
            st.markdown(f"🧾 _Ingredients_: {row['Ingredients']}")
            st.markdown("---")
