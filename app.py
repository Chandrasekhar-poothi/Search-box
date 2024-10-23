from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model for embeddings
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Load the CSV file containing course data
df = pd.read_csv('sample_courses.csv')

# Ensure the CSV has columns 'title' and 'description'
# Create embeddings for the course descriptions
df['embedding'] = df['description'].apply(lambda x: model.encode(x, show_progress_bar=False))

# Function to search for the top 3 most relevant courses
def search_courses(query, df, model):
    query_embedding = model.encode([query], show_progress_bar=False)
    similarities = cosine_similarity(query_embedding, np.vstack(df['embedding'].values.tolist()))  # Convert to list
    top_3_indices = np.argsort(similarities[0])[-3:][::-1]  # Get top 3 indices in descending order
    return df.iloc[top_3_indices]

# Flask route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Flask route for handling search requests
@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')  # Get query from form data
    top_courses = search_courses(query, df, model)
    
    # Create a list of top 3 course titles to return as JSON
    top_courses_list = top_courses['title'].tolist()
    
    return jsonify(top_courses_list)

# Run Flask app
if __name__ == "__main__": 
    app.run(debug=True)
