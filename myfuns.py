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


def myIBCF(S, user_data):
    rec = list()  # list of recommended movie id
    #rec_ratings = list()  # list of predicted rating of recommended movie

    #S = pd.read_csv('cos_similarity_matrix.csv', index_col=0)
    movie_id = S.columns

    # for each row, keep top 30 and set the rest to NA
    S = S.to_numpy()
    #for i in range(3706):
        #S[i, (np.nan_to_num(S[i, :])).argsort()[:3676]] = np.nan

    pred = np.array([0.0] * 3706)
    user = np.nan_to_num(user_data)
    mask = user != 0

    for i in range(3706):
        if mask[i]:  # user already review the movie, set prediciton to 0 to exclude it from recommendation
            pred[i] = 0
            continue

        si = np.nan_to_num(S[i, :])
        total_si = np.sum(si * mask)

        if total_si == 0:  # user didn't review any top 30 similar movies, set predicion to 0 to exclude it from recommendation
            pred[i] = 0
        else:
            pred[i] = np.sum(si * user) / total_si

    # rec = movie_id[pred.argsort()[-10:]]
    num_zero = np.sum(pred[pred.argsort()[-10:]] == 0)

    if num_zero < 10:
        rec = movie_id[pred.argsort()[-(10 - num_zero):]].astype(int).to_list()[-1:-(11 - num_zero):-1]
        #rec_ratings = pred[pred.argsort()[-(10 - num_zero):]].tolist()[-1:-(11 - num_zero):-1]

        if len(rec) < 10:  # fewer than 10 predicitons are non-NA,
            sid = movie_id.get_loc(str(rec[0]))  # movies similar to top recommended movie
            for m in S[sid, (np.nan_to_num(S[sid, :])).argsort()[-1:-31:-1]]:
                if m not in rec:
                    rec.append(m)
                    if len(rec) == 10:
                        break

    return rec#, rec_ratings