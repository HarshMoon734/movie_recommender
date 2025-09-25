import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import re

df_movie = pd.read_csv('tmdb_5000_movies.csv')
df_credit = pd.read_csv('tmdb_5000_credits.csv')

df_movie['movie_id'] = df_movie['id']
df_movie = df_movie.drop(['id'], axis=1)

df = pd.merge(df_movie, df_credit, on='movie_id', how='left')

def parse_genre_string(genre_str):
    return json.loads(genre_str)

df['genres'] = df['genres'].apply(parse_genre_string)

def processing_name(thing):
    answer = []
    for i in range(0,len(thing)):
        answer.append(thing[i]['name'])
    return " ".join(answer)
    
processing_name(df['genres'].iloc[2])

df['genres'] = df['genres'].apply(processing_name)

df = df.drop(['homepage'], axis=1)

df['keywords'] = df['keywords'].apply(parse_genre_string)
df["keywords"] = df['keywords'].apply(processing_name)

df['overview'] = df['overview'].fillna('')

df['production_companies'] = df['production_companies'].apply(parse_genre_string)
df['production_companies'] = df['production_companies'].apply(processing_name)

df['production_countries'] = df['production_countries'].apply(parse_genre_string)
df['production_countries'] = df['production_countries'].apply(processing_name)

df = df[~df['release_date'].isnull() == True].reset_index()

def years_passed_since(date_str):
    start_date = datetime.strptime(date_str, "%Y-%m-%d")
    today = datetime.today()

    years = today.year - start_date.year
    if (today.month, today.day) < (start_date.month, start_date.day):
        years -= 1

    return years

df['release_date'] = df['release_date'].apply(years_passed_since)

df = df[~df['runtime'].isnull() == True].reset_index()

df['spoken_languages'] = df['spoken_languages'].apply(parse_genre_string)
df['spoken_languages'] = df['spoken_languages'].apply(processing_name)

df['tagline'] = df['tagline'].fillna("")

df = df.drop(["movie_id"], axis=1)

df['cast'] = df['cast'].apply(parse_genre_string)


def fix_cast(json_file):
    answer = []
    
    for i in range(0,len(json_file)):
        answer.extend([json_file[i]['name'],json_file[i]['character']])
        
    return " ".join(answer)

df['cast'] = df['cast'].apply(fix_cast)

df = df.drop(['crew'], axis=1)
df = df.drop(['level_0','index'], axis=1)

df['corpus'] = df['genres']+" "+df['keywords']+" "+df['original_language']+" "+df['overview']+" "+df['production_companies']+" "+df['production_countries']+" "+df['spoken_languages']+" "+df['status']+" "+df['tagline']+" "+df['title_x']+" "+df['title_y']+" "+df['cast']

df = df.drop(['genres','keywords','original_language','overview','production_companies','production_countries','spoken_languages','status','tagline','title_x','title_y','cast'], axis=1)

df_final = df[['original_title','corpus']]
df_final["title"] = df_final["original_title"]
df_final = df_final.drop(['original_title'], axis=1)

def fix_corpus(string):
    corpus = re.sub('[^a-zA-Z]'," ",string)
    corpus = corpus.lower()
    
    answer = []
    
    lemmatizer = WordNetLemmatizer()
    
    corpus = corpus.split(' ')
    
    for l in corpus:
        if l not in set(stopwords.words('english')):
            answer.append(lemmatizer.lemmatize(l))
        
    return " ".join(answer)

df_final['corpus'] = df_final['corpus'].apply(fix_corpus)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['corpus'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_final.index, index=df_final['title']).drop_duplicates()

def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        return f"'{title}' not found in the dataset."
    
    idx = indices[title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return df_final['title'].iloc[movie_indices].tolist()

print(recommend('Code 46'))


# saving cosine similarity array, df_final, and indices array, in short, downloading the data we need to make an action.py script, which can directly do what we need to get

np.save('cosine_sim.npy', cosine_sim)
cosine_sim = np.load('cosine_sim.npy')

df_final.to_pickle('df_final.pkl')

indices.to_pickle('indices.pkl')