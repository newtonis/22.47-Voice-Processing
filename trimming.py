import numpy as np

criteria = 2.0/100.0


def trimming(audio):# we make a  trim using a criteria 2/100
    ### TODO: Improve using numpy.max()

    #max_value = 0
    #for x in range(len(audio)):
    #    max_value = max(max_value, abs(audio[x]))

    max_value = max(np.max(audio), -np.min(audio))

    start_taking = 0
    end_taking = len(audio)

    for x in range(len(audio)):
        if abs(audio[x]) > criteria * max_value:
            start_taking = x
            break

    for x in range(len(audio)-1, -1, -1):
        if abs(audio[x]) > criteria * max_value:
            end_taking = x
            break

    return audio[start_taking: end_taking]


def execute_trimming(database):
    ans = dict()

    for key in database:
        ans[key] = []

        for x in range(len(database[key])):
            if len(database[key][x].shape) > 1: # corrupted data
                continue
            x_trimmed = trimming(database[key][x])
            ans[key].append(x_trimmed)

    return ans
