import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from scipy.signal import convolve

class KEstimator():
    def __init__(self, k_opt):
        self.k_opt = k_opt
        self.alpha_k = None

    def __lossFunction(self, k, X):
        '''
        Parameters 
        ----------
        scaled_data: matrix "
            scaled data. rows are samples and columns are features for clustering
        k: int
            current k for applying KMeans
        alpha_k: float
            manually tuned factor that gives penalty to the number of clusters
        Returns 
        -------
        scaled_inertia: float
            scaled inertia value for current k           
        '''
        # fit k-means
        inertia_0 = KMeans(n_clusters=1).fit(X).inertia_
        kmeans = KMeans(n_clusters=k).fit(X)
        scaled_inertia = kmeans.inertia_ / inertia_0 + self.alpha_k * k
        return scaled_inertia

    def __chooseBestKforKMeansParallel(self, X,  k_range):
        '''
        Parameters 
        ----------
        scaled_data: matrix 
            scaled data. rows are samples and columns are features for clustering
        k_range: list of integers
            k range for applying KMeans
        Returns 
        -------
        best_k: int
            chosen value of k out of the given k range.
            chosen k is k with the minimum scaled inertia value.
        results: pandas DataFrame
            adjusted inertia value for each k in k_range
        '''

        ans = Parallel(n_jobs=-1,verbose=10)(delayed(lambda k:self.__lossFunction(k, X=X))(k) for k in k_range)
        ans = list(zip(k_range,ans))
        results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]
        return best_k, results

    def __chooseAlphaValue(self, X):
        '''
        This function tries to find the optimal value of alpha based on derivatives

        Parameters 
        -----
        scaled_data (array): matrix, the scaled data. 
        k_opt (int): for a given image, this is the optimal k we know, it is a training to find the optimum value of alpha.  
        '''
        self.alpha_k = 0
        k_range = [self.k_opt - 1, self.k_opt, self.k_opt+1]
        _, results = self.__chooseBestKforKMeansParallel(k_range = k_range, X=X)
        derivative = convolve(results.to_numpy(), np.array([[-1],[1]]), mode='same')
        self.alpha_k = np.round(derivative.flatten()[1], 4)

    def fit_k(self, X):
        self.__chooseAlphaValue(X)
        return self
    
    def predict_k(self, X,  k_range = [2,3,4,5,6,7,8,9]):
        if self.alpha_k is None:
            print("Please fit first")
        else:
            return self.__chooseBestKforKMeansParallel(X=X, k_range=k_range)[0]
