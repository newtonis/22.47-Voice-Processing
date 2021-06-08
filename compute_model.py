import numpy as np
from random import randrange
from compute_features import *
from sklearn.mixture import GaussianMixture
import pickle

audios_rate = 30.0 / 100.0


def compute_model(target_speaker, data):
    ## compute gmm model for target speaker using 30/100 of its audios
    stacked_feature = None

    for x in range(len(data)):
        index = randrange(len(data))

        currently_fetched_feature = data[target_speaker][index]  # one feature extracted
        if x == 0:
            stacked_feature = currently_fetched_feature
        else:
            stacked_feature = np.vstack((stacked_feature, currently_fetched_feature))

    gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=500, n_init=3, verbose=1)
    gmm.fit(stacked_feature)  # generating the GMM model of the stacked features

    pickle.dump(gmm, open("models/" + target_speaker, 'wb'))

