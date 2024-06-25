import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

movies=pd.read_csv("movies.csv")
credits=pd.read_csv("credits.csv")

movies.head()

credits.head()

movies.shape

credits.shape

movies=movies.merge(credits, on ='title')

movies.head()

movies.shape

movies['original_language'].value_counts()

movies.columns

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.head()

movies.shape

movies.dtypes

movies.isnull()

movies.isnull().sum()

movies.dropna(inplace=True)

movies.isnull().sum()

movies.shape

movies.duplicated()

movies.duplicated().sum()

movies.head()

movies.iloc[0]['genres']

import ast

ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

def convert(text):
    l=[]
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l

movies['genres']=movies['genres'].apply(convert)

movies['genres']

movies.head()

movies['keywords']=movies['keywords'].apply(convert)

movies

def convert_cast(text):
    l=[]
    counter=0
    for i in ast.literal_eval(text):
        if counter < 3:
            l.append(i['name'])
        counter+=1
    return l

movies['cast']=movies['cast'].apply(convert_cast)

movies['cast']

movies

def fetch_convert(text):
    l=[]
    for i in ast.literal_eval(text):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l

movies['crew']=movies['crew'].apply(fetch_convert)

movies['crew']

movies.head()

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies

def remove_space(word):
    l=[]
    for i in word:
        l.append(i.replace(" ",""))
    return l

movies['cast']=movies['cast'].apply(remove_space).copy()

movies['crew']=movies['crew'].apply(remove_space).copy()
movies['keywords']=movies['keywords'].apply(remove_space).copy()
movies['genres']=movies['genres'].apply(remove_space).copy()

movies

movies['tags']=movies['genres']+movies['overview']+movies['cast']+movies['crew']+movies['keywords']

movies

movies['tags']

new_data=movies[['movie_id','title','tags']]

new_data

new_data['tags']=new_data['tags'].apply(lambda x: " ".join(x)).copy()

new_data

movies['tags']

new_data['tags']=new_data['tags'].apply(lambda x:x.lower()).copy()

new_data

new_data.shape

import nltk
from nltk.stem import PorterStemmer

ps=PorterStemmer()

def stems(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

new_data['tags']=new_data['tags'].apply(stems).copy()

new_data['tags']

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=5000,stop_words='english')

vector=cv.fit_transform(new_data['tags']).toarray()

vector

vector.shape

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vector)

similarity

similarity.shape

new_data[new_data['title']=='Spider-Man'].index[0]

def recommend(movie):
    index=new_data[new_data['title']==movie].index[0]
    distances=sorted(list(enumerate(similarity[index])),reverse=True, key= lambda x:x[1])
    for i in distances[1:6]:
        print(new_data.iloc[i[0]].title)

recommend('The Dark Knight Rises')

import pickle

pickle.dump(new_data,open('artificates/movie_list.pkl','wb'))
pickle.dump(similarity,open('artificates/movie_similarity.pkl','wb'))
