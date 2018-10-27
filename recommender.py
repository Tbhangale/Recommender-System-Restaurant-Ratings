from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import SVD, KNNBasic, NMF
from surprise.model_selection import KFold
import os
import numpy as np

file_path = os.path.expanduser('D:/Sem 2/Temporal and spatial data/HW4/restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

#3-fold data without shuffeling so results are comparable fo each fold
kf = KFold(n_splits=3, random_state=None, shuffle=False)

#-----------------------------------------------------------
#Part 1 - Algorithms Comparison
#-----------------------------------------------------------

algorithms = {
    'SVD':SVD(),
    'PMF':SVD(biased=False),
    'NMF':NMF(),
    'UCF':KNNBasic(sim_options = {'user_based': True} ),
    'ICF':KNNBasic(sim_options = {'user_based': False} )
}
for name, algo in algorithms.items():
    rmse = []
    mae = []
    i = 1
    print("Evaluating RMSE, MAE of algorithm",name)
    print("=======================================")
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)
        # Compute and print Root Mean Squared Error and MAE
        print("fold -",i)
        i = i + 1
        rmse.append(accuracy.rmse(predictions, verbose=True))
        mae.append(accuracy.mae(predictions, verbose=True))
    print("\nRMSE Mean: %.4f" % np.mean(rmse))
    print("MAE Mean: %.4f"% np.mean(mae))
    print()

#-------------------------------------------------------
#Part 2 - Similarity Comparison
#-------------------------------------------------------

similarities = {
    'UCF with MSD':KNNBasic(sim_options = {'name': 'MSD', 'user_based': True}),
    'ICF with MSD':KNNBasic(sim_options = {'name': 'MSD', 'user_based': False}),
    'UCF with cosine':KNNBasic(sim_options = {'name': 'cosine', 'user_based': True}),
    'ICF with cosine':KNNBasic(sim_options = {'name': 'cosine', 'user_based': False}),
    'UCF with pearson':KNNBasic(sim_options = {'name': 'pearson', 'user_based': True}),
    'ICF with pearson':KNNBasic(sim_options = {'name': 'pearson', 'user_based': False}),
}
for name, sim in similarities.items():
    rmse = []
    mae = []
    print("Evaluating algorithm",name)
    print("==============================================")
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        sim.fit(trainset)
        predictions = sim.test(testset)
        # Compute and print Root Mean Squared Error and MAE
        rmse.append(accuracy.rmse(predictions,verbose=False))
        mae.append(accuracy.mae(predictions,verbose=False))
    print("\nRMSE Mean: %.4f" % np.mean(rmse))
    print("MAE Mean: %.4f"% np.mean(mae))
    print()


#---------------------------------------------------
#Part 3 - Finding the optimal number of neighbours
#---------------------------------------------------
cf_types = ["UCF", "ICF"]
for cf in cf_types:
    print("Evaluating RMSE of algorithm", cf)
    print("=======================================")
    print("RMSE Mean for 100 values of k:")
    rmse_mean = []
    for k in list(range(1,100)):
        algo = KNNBasic(k=k, sim_options = {'name':'msd', 'user_based': cf == "UCF"})
        rmse = []
        for trainset, testset in kf.split(data):
            # train and test algorithm.
            algo.fit(trainset)
            predictions = algo.test(testset,verbose=False)
            # Compute and print Root Mean Squared Error
            rmse.append(accuracy.rmse(predictions,verbose=False))
        rmse_mean.append("%.4f" % np.mean(rmse))
    print(rmse_mean)
    print()
