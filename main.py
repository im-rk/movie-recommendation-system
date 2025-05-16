import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load datasets
movie = pd.read_csv("movies.csv")
rating = pd.read_csv("ratings.csv")

# Preprocess ratings data
final_Data = rating.pivot(index="movieId", columns="userId", values="rating").fillna(0)

# Remove noisy data
no_user_voted = rating.groupby("movieId")["rating"].agg("count")
no_movies_voted = rating.groupby("userId")["rating"].agg("count")

final_Data = final_Data.loc[no_user_voted[no_user_voted > 10].index, :]
final_Data = final_Data.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Prepare CSR matrix and fit KNN model
csr_data = csr_matrix(final_Data.values)
final_Data.reset_index(inplace=True)

knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Request model
class MovieQuery(BaseModel):
    movie_title: str

# API endpoint
@app.post("/recommend")
def recommend_movies(query: MovieQuery):
    movie_name = query.movie_title.lower()
    movie_list = movie[movie['title'].str.lower().str.contains(movie_name)]

    if len(movie_list) == 0:
        raise HTTPException(status_code=404, detail="Movie not found.")

    movie_idx = movie_list.iloc[0]['movieId']

    if movie_idx not in final_Data['movieId'].values:
        raise HTTPException(status_code=404, detail="Movie not found in filtered data.")

    row_index = final_Data[final_Data['movieId'] == movie_idx].index[0]
    distances, indices = knn.kneighbors(csr_data[row_index], n_neighbors=11)

    recommended = []
    for i in range(1, len(distances[0])):  # Skip the input movie itself
        rec_movie_id = final_Data.iloc[indices[0][i]]['movieId']
        title = movie[movie['movieId'] == rec_movie_id]['title'].values[0]
        recommended.append({
            "title": title,
            "similarity": round(float(distances[0][i]), 3),
            "id":rec_movie_id
        })

    return {"recommendations": recommended}
