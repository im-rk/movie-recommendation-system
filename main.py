import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, Form  
from fastapi.responses import HTMLResponse  

# Creating FastAPI instance
app = FastAPI()  

"""##IMPORTING THE DATASET"""

movie=pd.read_csv(r"C:\Users\ramku\OneDrive\Documents\credit_risk_dataset.csv[1]\movies.csv")
rating=pd.read_csv(r"C:\Users\ramku\OneDrive\Documents\credit_risk_dataset.csv[1]\ratings.csv")

movie.head()

rating.head()

"""##EXTRACT UNIQUE MOVIE AND USER ID

"""

final_Data=rating.pivot(index="movieId",columns="userId",values="rating")
final_Data.head()

final_Data.fillna(0,inplace=True)
final_Data.head()

"""##REMOVING NOISE

"""

no_user_voted=rating.groupby("movieId")["rating"].agg("count")
no_movies_voted=rating.groupby("userId")["rating"].agg("count")

final_Data=final_Data.loc[no_user_voted[no_user_voted>10].index,:]

final_Data=final_Data.loc[:,no_movies_voted[no_movies_voted>50].index]

print(final_Data)

"""##CSR MATRIX"""

csr_data=csr_matrix(final_Data.values)
final_Data.reset_index(inplace=True)

"""##KNN MODEL

"""

knn=NearestNeighbors(metric="cosine",algorithm="brute",n_neighbors=20,n_jobs=-1)
knn.fit(csr_data)

"""##MOVIE RECOMMENDATION

"""

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movie[movie['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_Data[final_Data['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_Data.iloc[val[0]]['movieId']
            idx = movie[movie['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movie.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"

@app.get("/", response_class=HTMLResponse)
def get_form():
    html_content = """
    <html>
        <head><title>FastAPI Movie Recommendation</title></head>
        <body>
            <h2>Enter Movie Name</h2>
            <form action="/submit/" method="post">
                <input type="text" name="movie_name" required>  <!-- Movie name input -->
                <button type="submit">Get Recommendations</button>  <!-- Submit button -->
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)  

@app.post("/submit/")
def submit_data(movie_name: str = Form(...)):  
    recommendations = get_movie_recommendation(movie_name)  
    html_content = f"""
    <html>
        <head><title>FastAPI Movie Recommendation</title></head>
        <body>
            <h2>Movie Recommendations for "{movie_name}"</h2>  <!-- Displaying the movie name -->
            <ul>
    """

    if isinstance(recommendations, pd.DataFrame):  
        for _, rec in recommendations.iterrows():
            html_content += f"<li><b>{rec['Title']}</b> - Similarity Score: {rec['Distance']}</li>"

    else:
        html_content += f"<li>{recommendations}</li>"  

    html_content += """
            </ul>
            <br>
            <a href="/">Back</a>  <!-- Link to go back to the input form -->
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)  
