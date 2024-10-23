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

# Add CSS to set a background image
st.markdown(
    """
    <style>
    body {
        background-image: url("static/background.jpg");  /* Path to your local image */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }
    .container {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 30px;
        border-radius: 10px;
        max-width: 600px;
        width: 100%;
        margin: 0 auto;
        text-align: center;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<div class="container"><h1>Smart Course Recommendation Tool</h1>', unsafe_allow_html=True)

# Form for searching courses
with st.form(key='search_form'):
    query = st.text_input('Enter course keywords:')
    submit_button = st.form_submit_button(label='Search')

# Check if the form is submitted
if submit_button and query:
    # Perform the search
    top_courses = search_courses(query, df, model)

    # Display the results
    st.markdown('<h2>Top Course Recommendations:</h2>', unsafe_allow_html=True)
    for index, course in enumerate(top_courses['title']):
        st.write(f"{index + 1}. {course}")

# Close the container div
st.markdown('</div>', unsafe_allow_html=True)
