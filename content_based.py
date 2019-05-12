import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
# Reading movies file
movies = pd.read_csv('movies.csv', usecols=['movie_id', 'title', 'genres'])

# Break up the big genre string into a string array
movies['genres'] = movies['genres'].str.split('|')



# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')

#creating term frequency matrix of genres documents of each movie
vectorizer = CountVectorizer(min_df=0, stop_words=None, strip_accents='ascii')
docs_tf = vectorizer.fit_transform(movies['genres'])

#creating TFIDF matrix of genres documents of each movie
transformer = TfidfTransformer(smooth_idf = False)
tfidf = transformer.fit_transform(docs_tf.toarray())

#finding cosine value between vectors of obtained by TFIDF matrix and storing in cosine_sim matrix
cosine_sim = linear_kernel(tfidf.toarray(), tfidf.toarray())


# titles is array of movie titles
titles = movies['title']

# indices stores indice postion of movie 
indices = pd.Series(movies.index, index=movies['title'])


# Function that get movie recommendations based on the cosine similarity values of movie genres
def recommendations_based_on_genre(title):
    #finds the indices of movie title
    idx = indices[title]
    #finds the row in cosine_sim matrix for given index
    sim_values = list(enumerate(cosine_sim[idx]))
    #sorting the row based on cosine values
    sim_values = sorted(sim_values, key=lambda x: x[1], reverse=True)
    #taking first 20 values of sorted one
    sim_values = sim_values[1:11]
    #finding movie indices of first 20 movies
    movie_indices = [i[0] for i in sim_values]
    #finding name of movie titles using movie_indices
    return titles.iloc[movie_indices]

#gives the recommendation for input movie

movie_name = raw_input("Give the name of movie : ")


ind = -1
for x in movies['title']:
    if x == movie_name:
        ind = indices[movie_name]

if ind != -1:
    print recommendations_based_on_genre(movie_name).head(10)
else:
    print "Given name is not in dataset"