import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load dataset
file_path = 'articles recommendation dataset.csv'  # Ensure this file is in your Replit project
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Display the first few rows
print("Dataset preview:")
print(data.head())

# Preprocess data
print("\nChecking for missing values:")
print(data.isnull().sum())

# Fill missing values
data.fillna('', inplace=True)

# Content-Based Filtering
print("\nApplying Content-Based Filtering...")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Article'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    idx = data.index[data['Title'] == title].tolist()
    if not idx:
        return "Article not found."
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar items
    movie_indices = [i[0] for i in sim_scores]
    return data['Title'].iloc[movie_indices]

print("\nSample content-based recommendations for 'Best Books to Learn Data Analysis':")
print(get_content_based_recommendations('Best Books to Learn Data Analysis'))

# Save the content-based model
joblib.dump(tfidf, 'content_based_model.pkl')
joblib.dump(cosine_sim, 'cosine_similarity_matrix.pkl')
