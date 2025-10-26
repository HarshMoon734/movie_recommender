# Movie Recommender System (Content-Based Filtering)

This project builds a **content-based movie recommender system** using textual and metadata features from the **TMDB 5000 Movies and Credits** datasets.  
It processes movie metadata, creates a combined text representation for each movie, and uses **TF-IDF vectorization** with **cosine similarity** to recommend similar movies.

---

## Overview

The system extracts and processes multiple attributes such as genres, keywords, production companies, cast, and more to generate a detailed text corpus for each movie.  
Using TF-IDF and cosine similarity, it measures how similar two movies are based on their metadata and provides a ranked list of similar titles for any given movie.

---

## Requirements

Install the necessary dependencies before running the scripts:

```bash
pip install numpy pandas scikit-learn nltk matplotlib
```

Download the NLTK resources used for text cleaning and lemmatization:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## How It Works

1. **Dataset Loading**
   - Loads two CSV files:
     - `tmdb_5000_movies.csv`
     - `tmdb_5000_credits.csv`
   - Merges them on `movie_id` to form a unified dataset.

2. **Data Cleaning and Parsing**
   - Parses JSON-like string columns (`genres`, `keywords`, `production_companies`, etc.) into Python objects using `json.loads()`.
   - Extracts relevant text attributes (e.g., genre names, company names, language, etc.).
   - Drops irrelevant or null columns.

3. **Feature Engineering**
   - Combines selected attributes into a single column `corpus`:
     ```
     corpus = genres + keywords + original_language + overview + 
              production_companies + production_countries + 
              spoken_languages + status + tagline + title_x + title_y + cast
     ```
   - Applies text preprocessing:
     - Removes non-alphabetic characters
     - Converts to lowercase
     - Removes English stopwords
     - Lemmatizes words

4. **TF-IDF Vectorization**
   - Converts the text corpus into numerical feature vectors using:
     ```python
     from sklearn.feature_extraction.text import TfidfVectorizer
     tfidf = TfidfVectorizer(stop_words='english')
     tfidf_matrix = tfidf.fit_transform(df['corpus'])
     ```

5. **Similarity Computation**
   - Calculates **cosine similarity** between all pairs of movies:
     ```python
     from sklearn.metrics.pairwise import cosine_similarity
     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
     ```

6. **Recommendation Function**
   - Given a movie title, finds the 10 most similar movies based on cosine similarity:
     ```python
     def recommend(title, cosine_sim=cosine_sim):
         if title not in indices:
             return f"'{title}' not found in the dataset."
         idx = indices[title]
         sim_scores = list(enumerate(cosine_sim[idx]))
         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
         movie_indices = [i[0] for i in sim_scores]
         return df_final['title'].iloc[movie_indices].tolist()
     ```

7. **Data Export**
   - Saves essential data for reuse without retraining:
     ```
     cosine_sim.npy
     df_final.pkl
     indices.pkl
     ```

---

## Running the Main Script

To process data and generate the necessary files:

```bash
python main.py
```

This will:
- Process and clean the dataset  
- Compute similarity scores  
- Save the model data (`.npy` and `.pkl` files)  

---

## Using the Recommender (`action.py`)

Once the main preprocessing is complete, you can use the **action script** to get recommendations directly without reprocessing:

```bash
python action.py
```

Example output:
```
['Gattaca', 'The Island', 'Minority Report', 'The Thirteenth Floor', 'Surrogates', 
 'I, Robot', 'Children of Men', 'Transcendence', 'The Adjustment Bureau', 'The Time Machine']
```

---

## Notes

- Ensure the following files are present in the project directory before running `action.py`:
  - `cosine_sim.npy`
  - `df_final.pkl`
  - `indices.pkl`
- The input title must exactly match one in the dataset (`df_final['title']`).
- You can adjust the number of recommended movies by changing the slice in:
  ```python
  sim_scores = sim_scores[1:11]
  ```
