import numpy as np
import pickle

## compute DET Curves for model


def test_model(model_id, data):
    x_values, y_values = [], [] ## empty values for det curves
    model = pickle.load(open(model_id, 'rb'))

    cutting_points = np.logspace(0, 100, 100)

    for key in data:
        for x in range(len(data[key])):
            scores = np.array(model.score(data[key][x]))
            print(scores)

            ans = scores.sum()

