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
    year_playtime_df = genre_df.groupby('posted year')['playtime_forever'].sum().reset_index()
    max_playtime_year = year_playtime_df.loc[year_playtime_df['playtime_forever'].idxmax(), 'posted year']
    
    return f"Genre: {genre}. The year of launching with more hours played is {int(max_playtime_year)}"

print(TopYearPlaytimeGenre("Action"))