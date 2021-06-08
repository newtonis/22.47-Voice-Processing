import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt

## compute DET Curves for model


def test_model(model_file, data):
    x_values, y_values = [], [] ## empty values for det curves
    model = pickle.load(open(model_file, 'rb'))

    #cutting_points = np.logspace(0, 100, 100)

    answer = dict()

    for key in data:
        #sc = []
        answer[key] = []
        for x in range(len(data[key])):
            scores = np.array(model.score(data[key][x]))
            ans = scores.sum()
            answer[key].append(ans)

        # listvalues = ""
        # sum = 0
        # for x in range(len(sc)):
        #     listvalues += "%0.2f " % sc[x]
        #     sum += sc[x]
        #
        # listvalues += ",sum = %0.2f" % sum

        #print("score for '%s': %s" % (key, listvalues))
    return answer

""" testing that all models are better with own speaker """

def test_all_models(data):
    modelfiles = glob.glob("models/*")

    for file in modelfiles:
        answer = test_model(file, data)

        max_score = -100000
        winner = None
        for key in answer:
            total_score = sum(answer[key]) / len(answer[key])
            if total_score > max_score:
                max_score = total_score
                winner = key

        print("winner for model %s is %s (%0.2f)"% (file, winner, max_score) )


""" Testing DET curve variyng threshold score for comparison of model"""

def plot_det_curve(model_id, data):

    model = pickle.load(open("models/" + model_id, "rb"))


    cutting_points = np.linspace(-40, 0, 70)

    x_axis = []
    y_axis = []

    for cutting_point in cutting_points:
        print("Computing cutting point %d" % cutting_point)

        false_positives = 0  # count of false positives
        false_negatives = 0  # count of false negatives
        total_tests = 0  # count for total tests

        for key in data:
            for x in range(len(data[key])):
                score = np.array(model.score(data[key][x])).sum()
                if score > cutting_point:
                    # is speaker of model1
                    result = True
                else:
                    # is not speaker of model1
                    result = False

                if result and key != model_id:
                    false_positives += 1
                if not result and key == model_id:
                    false_negatives += 1
                total_tests += 1

        x_axis.append(false_negatives / total_tests * 100)
        y_axis.append(false_positives / total_tests * 100)

    plt.xlabel("False negatives (%)")
    plt.ylabel("False positives (%)")

    plt.plot(x_axis, y_axis)
    plt.show()

