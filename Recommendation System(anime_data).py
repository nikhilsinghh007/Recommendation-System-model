import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
file_path = 'Anime_data.csv'  # Replace with the actual path to your Anime_data.csv file
try:
    anime_data = pd.read_csv(file_path)
    print("Dataset loaded successfully")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    raise

# Print the first few rows of the dataset
print("First few rows of the dataset:\n", anime_data.head())

# Handle missing values
anime_data['Genre'] = anime_data['Genre'].fillna("['Unknown']")
anime_data['Type'] = anime_data['Type'].fillna('Unknown')
anime_data['Producer'] = anime_data['Producer'].fillna("['Unknown']")
anime_data['Studio'] = anime_data['Studio'].fillna("['Unknown']")
anime_data['Source'] = anime_data['Source'].fillna('Unknown')
anime_data['Rating'] = anime_data['Rating'].fillna(anime_data['Rating'].median())
anime_data['ScoredBy'] = anime_data['ScoredBy'].fillna(anime_data['ScoredBy'].median())
anime_data['Popularity'] = anime_data['Popularity'].fillna(anime_data['Popularity'].median())
anime_data['Members'] = anime_data['Members'].fillna(anime_data['Members'].median())
anime_data['Episodes'] = anime_data['Episodes'].fillna(anime_data['Episodes'].median())

anime_data = anime_data.dropna(subset=['Synopsis', 'Aired'])

print("Missing values handled")
print("Data types after handling missing values:\n", anime_data.dtypes)
print("Summary of the dataset after handling missing values:\n", anime_data.describe(include='all'))

# Transform Genre information
anime_data['Genre'] = anime_data['Genre'].apply(lambda x: ast.literal_eval(x))

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(anime_data['Genre'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

anime_data = pd.concat([anime_data, genre_df], axis=1)
anime_data = anime_data.drop('Genre', axis=1)

print("Genre information transformed")
print("First few rows of the transformed dataset:\n", anime_data.head())

# Create a user-anime interaction matrix
interaction_matrix = anime_data.pivot_table(index='Anime_id', columns='Title', values='Rating').fillna(0)
print("Interaction matrix created with shape:", interaction_matrix.shape)
print("First few rows of the interaction matrix:\n", interaction_matrix.head())

# Compute cosine similarity
anime_similarity = cosine_similarity(interaction_matrix)
print("Cosine similarity computed with shape:", anime_similarity.shape)

# Convert to DataFrame
anime_similarity_df = pd.DataFrame(anime_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
print("Cosine similarity DataFrame created with shape:", anime_similarity_df.shape)
print("First few rows of the cosine similarity DataFrame:\n", anime_similarity_df.head())

# Recommendation function
def get_recommendations(anime_id, num_recommendations=5):
    if anime_id not in anime_similarity_df.index:
        print(f"Anime ID {anime_id} not found in the dataset")
        return pd.DataFrame(columns=['Anime_id', 'Title'])

    similar_animes = anime_similarity_df[anime_id].sort_values(ascending=False)
    top_animes = similar_animes.iloc[1:num_recommendations + 1].index
    return anime_data.loc[anime_data['Anime_id'].isin(top_animes), ['Anime_id', 'Title']]

# Example usage
anime_id_example = anime_data['Anime_id'].iloc[0]  # Use the first anime ID in the dataset
recommended_animes = get_recommendations(anime_id=anime_id_example, num_recommendations=5)
print("Recommended animes:\n", recommended_animes)

# Save the recommendation results to a CSV file
recommended_animes.to_csv('recommended_animes.csv', index=False)
print("Recommended animes saved to recommended_animes.csv")
