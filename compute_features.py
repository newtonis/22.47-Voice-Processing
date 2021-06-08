import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import json
import os


def compute_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)# feature is preprocessed
    #delta = self.__calculate_delta(mfcc_feature)
    #combined = np.hstack((mfcc_feature, delta))
    return mfcc_feature


def execute_feature_computing(data):
    ans = dict()

    for key in data:
        ans[key] = []
        for x in range(len(data)):
            ans[key].append(compute_features(data[key][x], 44100))

    return ans


def save_features(data):
    data_to_save = dict()
    for key in data:
        data_to_save[key] = []
        for x in range(len(data[key])):
            data_to_save[key].append(data[key][x].tolist())

    data_to_write = json.dumps(data_to_save)

    with open('data_saved/features.txt', 'w') as f:
        f.write(data_to_write)


def saved_features_exist():
    return os.path.exists('data_saved/features.txt')


def load_features():
    with open('data_saved/features.txt', 'r') as f:
        data = f.read()

    return json.loads(data)
