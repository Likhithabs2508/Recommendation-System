import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# === Load Data ===
ratings = pd.read_csv('D:/infotactml/ml-latest-small/ratings.csv')
movies = pd.read_csv('D:/infotactml/ml-latest-small/movies.csv')
data = pd.merge(ratings, movies, on='movieId')

# === Create user-item matrix ===
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# === Apply SVD for recommendations ===
svd = TruncatedSVD(n_components=20)
matrix_reduced = svd.fit_transform(user_item_matrix)
approx_matrix = np.dot(matrix_reduced, svd.components_)

# === Define hybrid recommendation function ===
def hybrid_recommend(user_id, user_item_matrix, approx_matrix, top_n=5):
    rated_movies = user_item_matrix.iloc[user_id]
    rated_indices = rated_movies[rated_movies > 0].index
    preds = approx_matrix[user_id]
    movie_scores = pd.Series(preds, index=user_item_matrix.columns)
    movie_scores = movie_scores.drop(labels=rated_indices)
    return movie_scores.sort_values(ascending=False).head(top_n)

# === Streamlit UI ===
st.title("Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=0, max_value=user_item_matrix.shape[0] - 1, step=1)

if st.button("Get Recommendations"):
    recs = hybrid_recommend(user_id, user_item_matrix, approx_matrix)
    st.write("Top Recommendations:")
    for i, movie in enumerate(recs.index):
        st.write(f"{i+1}. {movie}")
