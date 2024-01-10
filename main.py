from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

app=FastAPI(debug=True)



df = pd.read_csv('databases/all_data.csv')

@app.get('/TopYearPlaytimeGenre/')
def TopYearPlaytimeGenre(genre: str) -> dict:
    genre = genre.capitalize()
    genre_df = df[df[genre] == 1]
    year_playtime_df = genre_df.groupby('posted_year')['playtime_forever'].sum().reset_index()
    max_playtime_year = year_playtime_df.loc[year_playtime_df['playtime_forever'].idxmax(), 'posted_year']
    return {"Genre": genre, "The year of launching with more hours played is:": int(max_playtime_year)}


#-------------------------------------------------------------------------------#

#Get the user with more hours in a given genre, then a list of ther hours through the years


@app.get('/TopUserForGenre/')
def TopUserForGenre(genre: str) -> dict:
    genre = genre.capitalize()
    genre_df = df[df[genre] == 1]
    max_playtime_user = genre_df.loc[genre_df['playtime_forever'].idxmax(), 'user_id']
    year_playtime_df = genre_df.groupby('posted_year')['playtime_forever'].sum().reset_index()
    playtime_list = year_playtime_df.to_dict(orient='records')
    result = {
        "Usuario con más horas jugadas para Género " + genre: [max_playtime_user],
        "Horas jugadas": [playtime_list]}
    return result

#-----------------------------------------------------------#

@app.get('/UsersRecommend/')
def UsersRecommend(year: int) -> dict:
    df_filtrado = df[(df['release_year'] == year) & (df['recommend'] == True) & (df['sentiment_score'] == 2)]
    if df_filtrado.empty:
        return {"error": 'Valor no encontrado'}
    df_ordenado = df_filtrado.sort_values(by='sentiment_score', ascending=False)
    top_3_reseñas = df_ordenado["title"].value_counts()
    
    resultado = {
        "Puesto 1": top_3_reseñas.index[0],
        "Puesto 2": top_3_reseñas.index[1],
        "Puesto 3": top_3_reseñas.index[2]
    }
    return resultado


#--------------------------------------------------------------#

@app.get('/UsersNotRecommed/')

def UsersNotRecommend(year: int) -> dict:
    df_filtrado = df[(df['release_year'] == year) & (df['recommend'] == False) & (df['sentiment_score'] == 0)]
    if df_filtrado.empty:
        return {"error": 'Valor no encontrado'}
    df_ordenado = df_filtrado.sort_values(by='sentiment_score', ascending=False)
    top_3_reseñas = df_ordenado["title"].value_counts()
    
    resultado = {
        "Puesto 1": top_3_reseñas.index[0],
        "Puesto 2": top_3_reseñas.index[1],
        "Puesto 3": top_3_reseñas.index[2]
    }
    return resultado

#------------------------------------------------------------------#

@app.get('/sentiment_analysis/')
def sentiment_analysis(year: int) -> dict:
    filtered_df = df[df['posted_year'] == year]
    sentiment_counts = filtered_df['sentiment_score'].value_counts()
    result = {
        "Positive": int(sentiment_counts.get(0, 0)),
        "Neutral": int(sentiment_counts.get(1, 0)),
        "Negative": int(sentiment_counts.get(2, 0))
    }
    return result


#--------------------------------------------------------------------#
#Model to recomend games based on similar games


sample = df.drop_duplicates(subset=["item_id"])
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(2,2))
sample=sample.fillna("")

tdfid_matrix = tfidf.fit_transform(sample['genres'])
cosine_similarity = linear_kernel( tdfid_matrix, tdfid_matrix)

@app.get('/game_to_games_recomendation')
def recomendation(id_game: int):
    if id_game not in sample['item_id'].values:
        return {'message': 'Game Id doesnt exist'}
    
    # Getting the genres column of the game
    genres = sample.columns[18:42] 
    
    # Filtrate df to keep games with similar genres
    filtered_df = sample[(sample[genres] == 1).any(axis=1)]
    
    # Simility of cosene to calculate similirity genre-wise
    tdfid_matrix_filtered = tfidf.transform(filtered_df['genres'])
    cosine_similarity_filtered = linear_kernel(tdfid_matrix_filtered, tdfid_matrix_filtered)
    
    idx = sample[sample['item_id'] == id_game].index[0]
    s_cosine = list(enumerate(cosine_similarity_filtered[idx]))
    s_scores = sorted(s_cosine, key=lambda x: x[1], reverse=True)
    s_ind = [i for i, _ in s_scores[1:6]]
    s_games = filtered_df['title'].iloc[s_ind].values.tolist()
    
    return {'Recommended games': list(s_games)}