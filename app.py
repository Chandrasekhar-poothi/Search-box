import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

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

# Function to load HTML and CSS content
def load_css():
    with open(os.path.join('static', 'styles.css')) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_html():
    with open(os.path.join('templates', 'index.html'), 'r') as f:
        html_content = f.read()
    st.markdown(html_content, unsafe_allow_html=True)

# Streamlit app layout
st.title('Smart Course Recommendation Tool')

# Load CSS
load_css()

# Load HTML template
load_html()

# Input field for user query
query = st.text_input('Enter course keywords:')

# Submit button
if st.button('Search'):
    if query:
        # Perform search and display results
        top_courses = search_courses(query, df, model)
        
        st.write('Top 3 Recommended Courses:')
        for index, course in enumerate(top_courses['title']):
            st.write(f"{index + 1}. {course}")

