import numpy as np
import pandas as pd

cosine_sim = np.load('cosine_sim.npy')
df_final = pd.read_pickle('df_final.pkl')
indices = pd.read_pickle('indices.pkl')

def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        return f"'{title}' not found in the dataset."
    
    idx = indices[title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return df_final['title'].iloc[movie_indices].tolist()

concerened_movie = 'Code 46'

print(recommend(concerened_movie))