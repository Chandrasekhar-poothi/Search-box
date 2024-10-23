import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model for embeddings
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Load the CSV file containing course data
df = pd.read_csv('sample_courses.csv')

# Create embeddings for the course descriptions
df['embedding'] = df['description'].apply(lambda x: model.encode(x))

# Function to search for the top 3 most relevant courses
def search_courses(query, df, model):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, np.vstack(df['embedding'].values))
    top_3_indices = np.argsort(similarities[0])[-3:][::-1]  # Get top 3 indices in descending order
    return df.iloc[top_3_indices]

# Read external HTML and CSS files
def load_html_and_css():
    # Load HTML content
    with open('templates/index.html', 'r', encoding='utf-8') as html_file:
        html_content = html_file.read()

    # Load CSS content
    with open('static/styles.css', 'r', encoding='utf-8') as css_file:
        css_content = css_file.read()

    return html_content, css_content

# Streamlit app layout
html_content, css_content = load_html_and_css()

# Inject CSS into the Streamlit app
st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

# Display the HTML structure
st.markdown(html_content, unsafe_allow_html=True)

# Input field for user query using Streamlit widget (integrated into your HTML)
query = st.text_input('Enter your query:')

if query:
    # Perform search and display results
    top_courses = search_courses(query, df, model)
    st.markdown("<div id='results'><h2>Top 3 Recommended Courses:</h2>", unsafe_allow_html=True)
    for index, course in enumerate(top_courses['title']):
        st.markdown(f"<p class='result-title'>{index + 1}. {course}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
