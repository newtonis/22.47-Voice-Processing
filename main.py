from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import glob
from scipy import signal
from scipy.fft import fftshift
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from compute_features import *
from compute_model import *
from trimming import *
from testing_model import *


def load_database():
    output = dict()

    for actor_id in range(0, 25):
        speaker_name = "speaker " + str(actor_id)

        filename = "archive/Actor_%02d/*" % actor_id
       # print("Actor id = %02d" % actor_id)

        file_list = glob.glob(filename)

        for file in file_list:
            #print("file = ", file)
            samplerate, data = wavfile.read(file)

            if not str(speaker_name) in output:
                output[str(speaker_name)] = []

            output[str(speaker_name)].append(data)

    return output


if not saved_features_exist():
    print("Loading database...")
    data = load_database()

    print("Computing trimming...")
    data_trimmed = execute_trimming(data)

    keys = list(data_trimmed.keys())

    print("Running trimming test...")
    ## showing test trimmed data
    x = data_trimmed[keys[0]][0]

    plt.plot(x)
    plt.show()
    ## compute vectors

    print("Computing data features...")
    data_features = execute_feature_computing(data_trimmed)

    print("Saving features...")
    save_features(data_features)

    print("Showing data feature example...")
    plt.plot(data_features["speaker 1"][0])
    plt.show()
else:
    print("Loading features from data saved")
    data_features = load_features()


# print("Computing model for speakers")
# for key in data_features:
#     print("Computing model for speaker ", key)
#     compute_model(key, data_features)

#print("Testing model for speaker 1")
#test_all_models(data_features)

plot_det_curve("speaker 1", data_features)

#plt.plot(x[0])
#plt.show()
#z = compute_mfcc(x) # mfcc


## we make fft test
#x = files[key0[0]][0][50000:100000]

#features = extract_features(x, 44100)

