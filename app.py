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

# Streamlit app layout
st.title('Smart Course Recommendation Tool')

# Input field for user query
query = st.text_input('Enter your query:')

if query:
    # Perform search and display results
    top_courses = search_courses(query, df, model)
    st.write('Top 3 Recommended Courses:')
    for course in top_courses['title']:
        st.write(course)
