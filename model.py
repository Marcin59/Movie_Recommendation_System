import pandas as pd
import numpy as np
import os.path
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

class Model:
    """
    A class representing a Movie Recommendation System model.

    Attributes:
    - k (int): The number of clusters for user clustering.
    - preprocessed_users_file_name (str): The file name for storing preprocessed user data.
    - ratings (pd.DataFrame): DataFrame containing user ratings data. 
    - users (pd.DataFrame): DataFrame containing user data.
    - movies (pd.DataFrame): DataFrame containing movie data.

    Methods:
    - fit(movies: pd.DataFrame, update_users=False): Fits the model to the given data.
    - save_users(path): Saves the user data to a CSV file.
    - get_relevant_genres(threshold=0.0): Returns a list of relevant movie genres based on the data.
    - get_users(selected_genres): Returns a DataFrame of users and their genre-based ratings.
    - clusterUsers(): Performs user clustering using K-means algorithm.
    - predict(userID, movieID): Predicts the rating for a given user and movie.
    - predictDummy(userID, movieID): Predicts the rating for a given user and movie using a dummy approach.
    - predictByDummy(userID, movieID): Predicts the rating for a given user and movie using a dummy approach.
    - predictByCLuster(userID, movieID): Predicts the rating for a given user and movie based on user clusters.
    - getBaseline(test): Calculates the mean squared error for the dummy approach.
    - getMSE(test): Calculates the mean squared error for the model's predictions.
    """

    def __init__(self, ratings: pd.DataFrame, k=30, preprocessed_users_file_name="./data/usersTest.csv", users=None):
        """
        Initializes a Movie Recommendation System model.

        Parameters:
        - k (int): The number of clusters for user clustering. Default is 30.
        - users_file_name (str): The file name for storing user data. Default is "./data/usersTest.csv".
        """
        self.users: pd.DataFrame = users
        self.movies: pd.DataFrame
        self.ratings = ratings
        self.preprocessed_users_file_name = preprocessed_users_file_name
        self.k = k

    def fit(self, movies: pd.DataFrame, update_users=False):
        """
        Fits the model to the given data.

        Parameters:
        - movies (pd.DataFrame): DataFrame containing movie data. Must have 'movieId' and 'genres' columns. Genres should be in format 'genre1|genre2|...'.
        - update_users (bool): Whether to update the user data file. Default is False.
        """
        movies.genres = movies.genres.str.split('|')
        self.movies = movies
        if not os.path.isfile(self.preprocessed_users_file_name) or update_users:
            selected_genres = self.get_relevant_genres()
            self.users = self.get_users(selected_genres)
        else:
            self.users = pd.read_csv(self.preprocessed_users_file_name)
        self.save_users(self.preprocessed_users_file_name)
        self.clusterUsers()

    def save_users(self, path):
        """
        Saves the user data to a CSV file.

        Parameters:
        - path (str): The path to save the file.
        """
        self.users.to_csv(path, index=False)
    
    def get_relevant_genres(self, threshold=0.0):
        """
        Returns a list of relevant movie genres based on the data.

        Parameters:
        - threshold (float): The minimum percentage of movies a genre should be associated with to be considered relevant. Default is 0.0.

        Returns:
        - selected_genres (list): List of relevant movie genres.
        """
        genre_count = self.movies['genres'].explode().value_counts().reset_index()
        genre_count.columns = ['genre', 'count']
        genre_count = genre_count[genre_count["count"] > self.movies.size*threshold]
        selected_genres = genre_count['genre'].tolist()
        return selected_genres  
    
    def get_users(self, selected_genres):
        """
        Returns a DataFrame of users and their genre-based ratings.

        Parameters:
        - selected_genres (list): List of relevant movie genres.

        Returns:
        - users (pd.DataFrame): DataFrame of users and their genre-based ratings.
        """
        users = pd.DataFrame(self.ratings['userId'].unique(), columns=['userId'])
        df = pd.merge(self.movies, self.ratings, on='movieId')
        for genre in selected_genres:
            genre_ratings = df[df['genres'].apply(lambda x: genre in x)]
            users[f"{genre}"] = users.apply(lambda x: genre_ratings[genre_ratings['userId'] == x['userId']]['rating'].mean(), axis=1)
        return users
    
    def clusterUsers(self):
        """
        Performs user clustering using K-means algorithm.
        """
        kmeans = KMeans(n_clusters=self.k)
        self.users = self.users.fillna(self.users.mean())
        usersCluster = kmeans.fit(self.users.drop('userId', axis=1))
        self.users['cluster'] = usersCluster.labels_

    def predict(self, userID, movieID):
        """
        Predicts the rating for a given user and movie.

        Parameters:
        - userID: The ID of the user.
        - movieID: The ID of the movie.

        Returns:
        - rating (float): The predicted rating.
        """
        if "cluster" not in self.users.columns:
            raise ValueError("Model has not been fitted.")
        clusterResult = self.predictByCLuster(userID, movieID)
        if not np.isnan(clusterResult):
            return clusterResult
        dummyResult = self.predictByDummy(userID, movieID)
        if not np.isnan(dummyResult):
            return dummyResult
        return self.ratings.rating.mean()
    
    def predictDummy(self, userID, movieID):
        """
        Predicts the rating for a given user and movie using a dummy approach. Cannot return NaN.

        Parameters:
        - userID: The ID of the user.
        - movieID: The ID of the movie.

        Returns:
        - rating (float): The predicted rating.
        """
        dummyResult = self.predictByDummy(userID, movieID)
        if not np.isnan(dummyResult):
            return dummyResult
        return self.ratings.rating.mean()
    
    def predictByDummy(self, userID, movieID):
        """
        Predicts the rating for a given user and movie using a dummy approach.
        
        Parameters:
        - userID: The ID of the user.
        - movieID: The ID of the movie.

        Returns:
        - rating (float): The predicted rating.
        """
        return self.ratings.loc[self.ratings['movieId'] == movieID].rating.mean()

    def predictByCLuster(self, userID, movieID):
        """
        Predicts the rating for a given user and movie based on user clusters.

        Parameters:
        - userID: The ID of the user.
        - movieID: The ID of the movie.

        Returns:
        - rating (float): The predicted rating.
        """
        if userID not in self.users['userId'].values:
            return np.nan
        userCluster = self.users.loc[self.users['userId'] == userID, 'cluster'].values[0]
        usersInCluster = self.users.loc[self.users['cluster'] == userCluster, 'userId']
        ratings = self.ratings[(self.ratings['movieId'] == movieID) & (self.ratings['userId'].isin(usersInCluster))]
        return ratings['rating'].mean()
    
    def getBaseline(self, test: pd.DataFrame):
        """
        Calculates the mean squared error for the dummy approach.

        Parameters:
        - test (pd.DataFrame): DataFrame containing test data.

        Returns:
        - mse (float): The mean squared error.
        """
        test["prediction"] = test.apply(lambda x: self.predictDummy(x["userId"], x["movieId"]), axis=1)
        return mean_squared_error(y_true = test.rating, y_pred = test.prediction)

    def getMSE(self, test: pd.DataFrame):
        """
        Calculates the mean squared error for the model's predictions.

        Parameters:
        - test (pd.DataFrame): DataFrame containing test data. Must have 'userId', 'movieId', and 'rating' columns.

        Returns:
        - mse (float): The mean squared error.
        """
        test["prediction"] = test.apply(lambda x: self.predict(x["userId"], x["movieId"]), axis=1)
        return mean_squared_error(y_true = test.rating, y_pred = test.prediction)
