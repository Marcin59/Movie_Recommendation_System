import pandas as pd
import numpy as np

# This script is used to prepare the test data for the model by taking a subset of the original data
if __name__ == '__main__':
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')

    movies.to_csv('data/test/movies.csv')
    ratings = ratings.loc[ratings["userId"] < 10000]
    ratings.to_csv('data/test/ratings.csv')