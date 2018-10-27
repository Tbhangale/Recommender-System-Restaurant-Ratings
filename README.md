# Recommender Systems on Restaurant Ratings data

1. Recommender systems aim to predict the rating that a user will give for an item (e.g., a restaurant, a movie, a product, a Point of Interest).

  Surprise (http://surpriselib.com) is a Python package for developing recommender systems. To install
  Surprise, the easiest way is to use pip.
  Open your console:
  $ pip install numpy
  $ pip install scikit-surprise

2. Download an experimental dataset “restaurant_ratings.txt”: Files/Data/restaurant_ratings.txt. Load data from “restaurant_ratings.txt” with line format: 'user item rating timestamp'.

3. MAE and RMSE are two famous metrics for evaluating the performances of a recommender system.

4. Split the data for 3-folds cross-validation, and compute the MAE and RMSE of the SVD
(Singular Value Decomposition) algorithm, PMF (Probabilistic Matrix Factorization) algorithm, NMF
(Nonnegative Matrix Factorization) algorithm, User based Collaborative Filtering algorithm, Item based
Collaborative Filtering algorithm

5. Compare the performances of SVD, PMF, NMF, UCF, ICF on fold-1, fold-2, fold-3 with respect to RMSE and MAE. Since data.split(n_folds=3)randomly split the data into 3 folds, please make sure you test the five algorithms on the same fold-1 so the results are comparable.

6. Examine how the cosine, MSD (Mean Squared Difference), and Pearson similarities impact the performances of User based Collaborative Filtering and Item based Collaborative Filtering.

7. Examine how the number of neighbors impacts the performances of User based Collaborative Filtering or Item based Collaborative Filtering? Plot the results.
