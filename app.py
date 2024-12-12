import streamlit as st
import pandas as pd
from typing import Dict
from myfuns import genres, get_displayed_movies, get_popular_movies, get_recommended_movies

# def get_movie_card(movie, with_rating=False):
#     """Generates a movie card with an optional rating input."""
#     st.image(
#         f"https://liangfgithub.github.io/MovieImages/{movie.movie_id}.jpg?raw=true",
#         width=150,
#     )
#     st.markdown(f"**{movie.title}**")

#     if with_rating:
#         return st.radio(
#             f"Rate {movie.title}",
#             [None, 1, 2, 3, 4, 5],
#             horizontal=True,
#             key=f"rating_{movie.movie_id}",
#         )

# def show_genre_recommendation_page():
#     st.title("Select a Genre")
#     genre = st.selectbox("Choose a genre", genres)

#     if genre:
#         st.subheader(f"Popular movies in {genre}")
#         movies = get_popular_movies(genre)

#         for idx, movie in movies.iterrows():
#             with st.container():
#                 get_movie_card(movie)

# def show_collaborative_page():
#     st.title("Rate Movies for Recommendations")

#     # Display movies for rating
#     movies = get_displayed_movies()
#     ratings = {}

#     st.subheader("Rate these movies")
#     for idx, movie in movies.iterrows():
#         with st.container():
#             rating = get_movie_card(movie, with_rating=True)
#             if rating:
#                 ratings[movie.movie_id] = rating

#     # Show recommendations button
#     if st.button("Get Recommendations"):
#         if not ratings:
#             st.warning("Please rate at least one movie to get recommendations!")
#             return

#         # Fetch recommendations
#         st.subheader("Your Recommendations")
#         recommended_movies = get_recommended_movies(ratings)

#         for idx, movie in recommended_movies.iterrows():
#             with st.container():
#                 get_movie_card(movie)




def get_movie_card(movie, with_rating=False):
    """Generates a movie card with an optional rating input."""
    st.image(
        f"https://liangfgithub.github.io/MovieImages/{movie.movie_id}.jpg?raw=true",
        width=150,
    )
    st.markdown(f"**{movie.title}**")

    if with_rating:
        return st.radio(
            f"Rate {movie.title}",
            [None, 1, 2, 3, 4, 5],
            horizontal=True,
            key=f"rating_{movie.movie_id}",
        )

def display_movies_in_grid(movies, with_rating=False):
    """Displays movies in a grid with 5 movies per row."""
    cols_per_row = 5
    num_movies = len(movies)
    rows = (num_movies + cols_per_row - 1) // cols_per_row  # Calculate number of rows

    for i in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < num_movies:
                movie = movies.iloc[idx]
                with cols[j]:
                    get_movie_card(movie, with_rating=with_rating)

def show_genre_recommendation_page():
    st.title("Select a Genre")
    genre = st.selectbox("Choose a genre", genres)

    if genre:
        st.subheader(f"Popular movies in {genre}")
        movies = get_popular_movies(genre)
        display_movies_in_grid(movies)

def show_collaborative_page():
    st.title("Rate Movies for Recommendations")

    # Display movies for rating
    movies = get_displayed_movies()
    ratings = {}

    st.subheader("Rate these movies")
    display_movies_in_grid(movies, with_rating=True)
    for idx, movie in movies.iterrows():
            if rating:
                ratings[movie.movie_id] = rating

    # Show recommendations button
    if st.button("Get Recommendations"):
        if not ratings:
            st.warning("Please rate at least one movie to get recommendations!")
            return

        # Fetch recommendations
        st.subheader("Your Recommendations")
        recommended_movies = get_recommended_movies(ratings)
        display_movies_in_grid(recommended_movies)


def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")

    # Sidebar Navigation
    st.sidebar.title("Movie Recommender")
    page = st.sidebar.radio(
        "Navigation", ["System 1 - Genre", "System 2 - Collaborative"]
    )

    if page == "System 1 - Genre":
        show_genre_recommendation_page()
    elif page == "System 2 - Collaborative":
        show_collaborative_page()

if __name__ == "__main__":
    main()
