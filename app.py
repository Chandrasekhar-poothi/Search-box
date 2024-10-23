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

# Define the HTML and CSS content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Search Courses</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-image: url('/static/background.jpg'); /* Example background */
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #fff;
        }
        .container {
            background-color: rgba(255, 255, 255, 0.3);
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
            max-width: 600px;
            width: 100%;
            text-align: center;
        }
        h1 {
            color: #0d0101;
            font-size: 3rem;
            margin-bottom: 20px;
        }
        input {
            width: 100%;
            padding: 10px;
            margin: 15px 0;
            border: 1px solid #8e1414;
            border-radius: 5px;
            font-size: 1rem;
            box-sizing: border-box;
        }
        button {
            padding: 10px 20px;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
        }
        button:hover {
            background-color: #218838;
        }
        #results {
            margin-top: 20px;
            font-size: 1rem;
            text-align: left;
            background-color: #1c5d2b;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Search for Courses</h1>
        <form id="searchForm">
            <input type="text" id="query" name="query" placeholder="Enter course keywords" required>
            <button type="submit">Search</button>
        </form>
        
        <div id="results"></div>
    </div>

    <script>
        const form = document.getElementById('searchForm');
        const resultsDiv = document.getElementById('results');
        
        form.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const query = document.getElementById('query').value;
            
            fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    query: query
                })
            })
            .then(response => response.json())
            .then(data => {
                // Display the top 3 recommended courses
                resultsDiv.innerHTML = '<h2>Top Course Recommendations:</h2>';
                data.forEach((course, index) => {
                    resultsDiv.innerHTML += `<p>${index + 1}. ${course}</p>`;
                });
            })
            .catch(error => console.error('Error:', error));
        });
    </script>
</body>
</html>
"""

# Streamlit app layout
st.title('Smart Course Recommendation Tool')

# Render HTML and CSS using st.markdown
st.markdown(html_content, unsafe_allow_html=True)

# Input field for user query
query = st.text_input('Enter your query:')

if query:
    # Perform search and display results
    top_courses = search_courses(query, df, model)
    st.write('Top 3 Recommended Courses:')
    for course in top_courses['title']:
        st.write(course)
