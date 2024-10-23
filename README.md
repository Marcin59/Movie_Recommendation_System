# Movie Recommendation System using K-means Algorithm

## Project Overview

This project implements a movie recommendation system utilizing the K-means clustering algorithm. The system is designed to predict users' movie ratings based on their viewing history and clustering them into groups with similar preferences. Using the MovieLens dataset, the system helps film enthusiasts discover movies they may enjoy without relying solely on traditional review scores.

## Features

- **K-means Clustering**: Groups users based on their rating patterns for different movie genres.
- **Personalized Recommendations**: Predicts a user's rating for a movie by analyzing the ratings of similar users within the same cluster.
- **Scalable Design**: Can handle large datasets like MovieLens, which contains millions of ratings and thousands of movies.
- **Modular Structure**: The algorithm can be easily adapted for other types of recommendation systems (e.g., books, music).

## Algorithm

The K-means algorithm is applied to user ratings data, specifically targeting the movie genres that a user has rated. The process is as follows:

1. **Clustering Users**: Users are clustered based on their average ratings across different movie genres.
2. **Prediction**: To predict a user's rating for a movie, the system looks at the user's cluster and averages the ratings of that movie from users in the same cluster.
3. **Evaluation**: The system was tested using a baseline of an average movie rating and compared to the K-means approach. The K-means model showed improved accuracy with a lower Mean Squared Error (MSE).

## Results

- The K-means model achieved the best performance with **4 clusters** when using the smaller dataset, with an MSE of **0.862**.
- For the full dataset, the optimal performance was achieved with **12 clusters**, resulting in an MSE of **0.77**.
- The results indicate that increasing the number of clusters beyond a certain point does not significantly improve the performance.

## Team members:

- Marcin Kapiszewski (code) - 156048
- Maciej Janicki (presentation) - 156073
- Adam Tomys (report) - 156057
- Marcel Rojewski (chief) - 156059

## How to reproduce the experiment:

- Download required packages using requirements.txt
- To use our model you need to pull data from dvc
- In load_and_predict.ipynb is shown how to predict grade for specific movie and users and how to get mse for a batch of them
