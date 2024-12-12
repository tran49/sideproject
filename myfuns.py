import pandas as pd
import numpy as np
#import requests

# # Define the URL for movie data
# myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
#
# # Fetch the data from the URL
# response = requests.get(myurl)
#
# # Split the data into lines and then split each line using "::"
# movie_lines = response.text.split('\n')
# movie_data = [line.split("::") for line in movie_lines if line]
#
# # Create a DataFrame from the movie data
# movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
# movies['movie_id'] = movies['movie_id'].astype(int)

# genres = list(
#     sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
# )
movies = pd.read_csv('movies.dat', sep='::', engine='python',
                     encoding='ISO-8859-1', header=None)
movies.columns = ['movie_id', 'title', 'genres']

genres = ['Action', 'Adventure', 'Animation', "Children's", "Comedy",  "Crime",
         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
         'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

genre_top10 = pd.read_csv('genre_top10.csv', index_col=0)

cos_similarity_matrix = pd.read_csv('cos_similarity_matrix_sorted.csv', index_col=0)

popularity = pd.read_csv('popularity.csv', index_col = 0)
def get_displayed_movies():
    return movies.head(100)


def get_recommended_movies(new_user_ratings):
    #return movies.head(10)
    #print(new_user_ratings)
    movie_id = cos_similarity_matrix.columns
    movie_id_list = movie_id.to_list()

    unew = np.array([0.0] * 3706)
    #unew[movie_id_list.index(str(1613))] = 5.0
    #unew[movie_id_list.index(str(1755))] = 4.0
    for k, v in new_user_ratings.items():
        unew[movie_id_list.index(str(k))] = float(v)

    rec = myIBCF(cos_similarity_matrix, unew)

    ret = pd.DataFrame({'movie_id': rec,
                        'title': [movies[movies.movie_id == id]['title'].values[0] for id in rec]})
    return ret


def get_popular_movies(genre: str):
    rec = genre_top10[genre].to_list()
    ret = pd.DataFrame({'movie_id': rec,
                        'title': [movies[movies.movie_id == id]['title'].values[0] for id in rec]})
    return ret
    #return movies[movies.movie_id.isin(genre_top10[genre])]
    #return genre_top10[genre].tolist()


# def myIBCF(S, user_data):
#     rec = list()  # list of recommended movie id
#     #rec_ratings = list()  # list of predicted rating of recommended movie

#     #S = pd.read_csv('cos_similarity_matrix.csv', index_col=0)
#     movie_id = S.columns

#     # for each row, keep top 30 and set the rest to NA
#     S = S.to_numpy()
#     #for i in range(3706):
#         #S[i, (np.nan_to_num(S[i, :])).argsort()[:3676]] = np.nan

#     pred = np.array([0.0] * 3706)
#     user = np.nan_to_num(user_data)
#     mask = user != 0

#     for i in range(3706):
#         if mask[i]:  # user already review the movie, set prediciton to 0 to exclude it from recommendation
#             pred[i] = 0
#             continue

#         si = np.nan_to_num(S[i, :])
#         total_si = np.sum(si * mask)

#         if total_si == 0:  # user didn't review any top 30 similar movies, set predicion to 0 to exclude it from recommendation
#             pred[i] = 0
#         else:
#             pred[i] = np.sum(si * user) / total_si

#     # rec = movie_id[pred.argsort()[-10:]]
#     num_zero = np.sum(pred[pred.argsort()[-10:]] == 0)

#     if num_zero < 10:
#         rec = movie_id[pred.argsort()[-(10 - num_zero):]].astype(int).to_list()[-1:-(11 - num_zero):-1]
#         #rec_ratings = pred[pred.argsort()[-(10 - num_zero):]].tolist()[-1:-(11 - num_zero):-1]

#         if len(rec) < 10:  # fewer than 10 predicitons are non-NA,
#             sid = movie_id.get_loc(str(rec[0]))  # movies similar to top recommended movie
#             for m in S[sid, (np.nan_to_num(S[sid, :])).argsort()[-1:-31:-1]]:
#                 if m not in rec:
#                     rec.append(m)
#                     if len(rec) == 10:
#                         break

#     return rec#, rec_ratings


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

    for i in range(S.shape[0]):  # Loop through all movies (rows in S)
        # Skip movies already rated by the new user
        if not np.isnan(new_user[i]):
            continue

        # Identify S(i) = {l: Sil ≠ NA}, where l ≠ NA in S[i, :]
        valid_similarities = ~np.isnan(S[i, :])
        weights = S[i, valid_similarities]
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

        # Exclude movies already rated by the user
        already_rated = {movie_ids[i] for i, rating in enumerate(new_user) if not np.isnan(rating)}
        remaining_popular_movies = top_movies[~top_movies['MovieID'].isin(already_rated)]

        # Fill the gap with popular movies, adding their average rating
        for _, row in remaining_popular_movies.iterrows():
            if len(recommendations) >= 10:
                break
            recommendations.append((f"m{row['MovieID']}", f"{row['avg_rating']:.7f}"))

    return recommendations
