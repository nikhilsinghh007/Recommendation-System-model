import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
file_path = 'fashion_products.csv'  # Update to your actual file path
try:
    fashion_data = pd.read_csv(file_path)
    print("Dataset loaded successfully")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    raise

# Print the first few rows of the dataset
print("First few rows of the dataset:\n", fashion_data.head())

# Handle missing values (if necessary)
fashion_data['Brand'] = fashion_data['Brand'].fillna('Unknown')
fashion_data['Category'] = fashion_data['Category'].fillna('Unknown')
fashion_data['Color'] = fashion_data['Color'].fillna('Unknown')
fashion_data['Size'] = fashion_data['Size'].fillna('Unknown')

fashion_data['Price'] = fashion_data['Price'].fillna(fashion_data['Price'].median())
fashion_data['Rating'] = fashion_data['Rating'].fillna(fashion_data['Rating'].median())

print("Missing values handled")
print("Data types after handling missing values:\n", fashion_data.dtypes)
print("Summary of the dataset after handling missing values:\n", fashion_data.describe(include='all'))

# Create a user-product interaction matrix
interaction_matrix = fashion_data.pivot_table(index='User ID', columns='Product ID', values='Rating').fillna(0)
print("Interaction matrix created with shape:", interaction_matrix.shape)
print("First few rows of the interaction matrix:\n", interaction_matrix.head())

# Compute cosine similarity
user_similarity = cosine_similarity(interaction_matrix)
print("Cosine similarity computed with shape:", user_similarity.shape)

# Convert to DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
print("Cosine similarity DataFrame created with shape:", user_similarity_df.shape)
print("First few rows of the cosine similarity DataFrame:\n", user_similarity_df.head())

# Recommendation function
def get_recommendations(user_id, num_recommendations=5):
    if user_id not in user_similarity_df.index:
        print(f"User ID {user_id} not found in the dataset")
        return pd.DataFrame(columns=['User ID', 'Product ID'])

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    top_users = similar_users.iloc[1:num_recommendations + 1].index
    # Find products rated by top similar users
    top_products = fashion_data[fashion_data['User ID'].isin(top_users)]['Product ID'].value_counts().index
    return pd.DataFrame({'User ID': [user_id]*len(top_products), 'Product ID': top_products})

# Example usage
user_id_example = fashion_data['User ID'].iloc[0]  # Use the first user ID in the dataset
recommended_products = get_recommendations(user_id=user_id_example, num_recommendations=5)
print("Recommended products:\n", recommended_products)

# Save the recommendation results to a CSV file
recommended_products.to_csv('recommended_products.csv', index=False)
print("Recommended products saved to recommended_products.csv")
