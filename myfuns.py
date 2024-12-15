import pandas as pd
import numpy as np

movies = pd.read_csv('movies.dat', sep='::', engine='python',
                     encoding='ISO-8859-1', header=None)
movies.columns = ['movie_id', 'title', 'genres']

genres = ['Action', 'Adventure', 'Animation', "Children's", "Comedy",  "Crime",
         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
         'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

genre_top10 = pd.read_csv('genre_top10.csv', index_col=0)

cos_similarity_matrix = pd.read_csv('cos_similarity_matrix_sorted.csv', index_col=0)

popularity = pd.read_csv('popularity.csv')

def get_displayed_movies():
    return movies.head(100)


def get_recommended_movies(new_user_ratings):
    movie_id = cos_similarity_matrix.columns
    movie_id_list = movie_id.to_list()

    unew = np.array([np.nan] * 3706)

    for k, v in new_user_ratings.items():
        unew[movie_id_list.index(str(k))] = float(v)

    rec = myIBCF(cos_similarity_matrix, unew)

    rec_movies = pd.DataFrame({'movie_id': rec,
                        'title': [movies[movies.movie_id.astype(str) == id]['title'].values[0] for id in rec]})
    return rec_movies


def get_popular_movies(genre: str):
    top10 = genre_top10[genre].to_list()
    popular_movies = pd.DataFrame({'movie_id': top10,
                        'title': [movies[movies.movie_id == id]['title'].values[0] for id in top10]})
    return popular_movies


def myIBCF(S, new_user):
    """
    Implements Item-Based Collaborative Filtering (IBCF) for a new user.

    Input:
        new_user (np.array): A 3706-by-1 vector containing ratings for 3706 movies by a new user.
                             Ratings should be integers 1, 2, 3, 4, or 5, and NA values as np.nan.
                             The order of the movies should match the rating matrix R.

    Output:
        recommendations (list): Top 10 recommended movies as tuples of (movie_id, predicted_rating).
    """
    predictions = []
    S_np = S.to_numpy()

    for i in range(S.shape[0]):  # Loop through all movies (rows in S)
        # Skip movies already rated by the new user
        if not np.isnan(new_user[i]):
            continue

        # Identify S(i) = {l: Sil ≠ NA}, where l ≠ NA in S[i, :]
        valid_similarities = ~np.isnan(S_np[i, :])
        weights = S_np[i, valid_similarities]
        rated_movies = new_user[valid_similarities]

        # Consider only movies rated by the new user
        rated_indices = ~np.isnan(rated_movies)
        weights = weights[rated_indices]
        ratings = rated_movies[rated_indices]

        # Skip if no valid ratings are available
        if len(ratings) == 0 or len(weights) == 0:
            continue

        # Compute prediction for movie i
        numerator = np.sum(weights * ratings)
        denominator = np.sum(weights)
        if denominator > 0:
            predicted_rating = round(numerator / denominator, 7)
            predictions.append((i, predicted_rating))

    # Sort predictions by rating in descending order
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Extract top 10 recommendations with actual movie IDs
    movie_ids = S.columns.tolist()
    recommendations = [(movie_ids[i], f"{predicted_rating:.7f}") for i, predicted_rating in predictions[:10]]

    # If fewer than 10 predictions, add popular movies based on average ratings
    if len(predictions) < 10:
        # System 1 popular movies
        popular_movies = popularity[popularity["review_count"] >= 2000]
        top_movies = popular_movies.sort_values("avg_rating", ascending=False).head(10)
        print(top_movies.columns)

        # Exclude movies already rated by the user
        already_rated = {movie_ids[i] for i, rating in enumerate(new_user) if not np.isnan(rating)}
        remaining_popular_movies = top_movies[~top_movies['MovieID'].isin(already_rated)]

        # Fill the gap with popular movies, adding their average rating
        for _, row in remaining_popular_movies.iterrows():
            if len(recommendations) >= 10:
                break
            print("row['MovieID'] : ", row['MovieID'])
            recommendations.append(int(row['MovieID']))
    return recommendations
