import streamlit as st
import pickle

# Load the pickled data
new_data = pickle.load(open('artificates/movie_list.pkl', 'rb'))
similarity = pickle.load(open('artificates/movie_similarity.pkl', 'rb'))

# Function to recommend movies
def recommend(movie):
    index = new_data[new_data['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(new_data.iloc[i[0]].title)
    return recommended_movies

# Streamlit UI
st.title('Movie Recommendation System')

# Dropdown for movie selection
movie_list = new_data['title'].values
selected_movie = st.selectbox('Select a movie:', movie_list)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.write(f'**You may also like:**')
    for movie in recommendations:
        st.write(f'- {movie}')
