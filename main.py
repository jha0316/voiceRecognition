import os
from pathlib import Path
#import numpy as np
# from extract_features import get_audio_features
# from age_model import lstm_age_model
# from utils import norm_multiple
# from file_io import get_data_files

#########################################################################################################################
import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "audio/temp.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
#########################################################################################################################
from scipy.signal import lfilter
from pitch_utils import get_pitch_magnitude, smooth
import numpy as np
import librosa
import numpy
import keras_metrics as km
from keras import Sequential
from keras.layers import LSTM, Dense

def normalize_data(input_data: np.array, means, std_dev):
    """
    normalize data by simple statistic data normalisation
    :param input_data: data to be normalized
    :param means:
    :param std_dev:
    :return: normalized data
    """
    norm_input_data = (input_data - means) / std_dev
    return norm_input_data

def norm_multiple(input_data: np.array, mean_paths, stddev_paths):
    """
    normalize data after multiple means, stddevs files

    :param input_data: data to be normalized
    :param mean_paths: list containing paths to the means
    :param stddev_paths: list containing paths to the stddevs
    :return: list of normalized data, where each is normalized
             after a pair of (mean, stddev)
    """

    norm_input_data = []

    for mean_path, stddev_path in zip(mean_paths, stddev_paths):
        means = np.load(mean_path)
        stddevs = np.load(stddev_path)

        input_copy = np.copy(input_data)
        norm_input = normalize_data(input_copy, means, stddevs)
        norm_input_data.append(norm_input)

    return norm_input_data

NUM_FEATURES = 41  # 39
def lstm_age_model(num_labels):
    model = Sequential()
    model.add(LSTM(128 * 2, input_shape=(35, NUM_FEATURES), return_sequences=True, dropout=0.3))
    model.add(LSTM(128 * 2, dropout=0.3))
    model.add(Dense(128 * 2, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', km.categorical_precision()])
    return model

def extract_max(pitches, magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(numpy.max(pitches[:, i]))
        new_magnitudes.append(numpy.max(magnitudes[:, i]))
    return numpy.asarray(new_pitches), numpy.asarray(new_magnitudes)


def smooth(x, window_len=11, window='hanning'):
    if window_len < 3:
        return x
    s = numpy.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[window_len:- window_len + 1]


def analyse(y, sr, fmin, fmax):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, S=None, fmin=fmin,
                                                fmax=fmax, threshold=0.75)
    shape = numpy.shape(pitches)
    pitches, magnitudes = extract_max(pitches, magnitudes, shape)
    return pitches, magnitudes


def get_pitch_magnitude(audio_data_path, sample_rate):

    duration = librosa.get_duration(filename=str(audio_data_path))
    y, sr = librosa.load(audio_data_path, sr=sample_rate, duration=duration, mono=True)
    pitches, magnitudes = analyse(y, sr, fmin=80, fmax=250)

    return y, pitches, magnitudes

def write_to_file_features(output_file, features):
    """
    writes a data sample(matrix) to file
    write whole dataset to file:
        for i in range(dataset.shape[0]):
            write_to_file_features("example.txt", dataset[i])
    :param output_file: file to write to
    :param features: data sample
    """
    with open(output_file, 'a+') as file_stream:
        for f in features:
            for el in f:
                file_stream.write(str(el))
                file_stream.write(",")
        file_stream.write('\n')

def write_to_file_labels(output_file, vector):
    """
    write elements of a 1d vector to file
    :param output_file: output file
    :param vector: data to be written
    """
    with open(output_file, 'w+') as file:
        for item in vector:
            file.write(str(item))
            file.write('\n')

def features_from_file(input_file, num_features=20):
    """
    extract mfcc features from file
    :param input_file: feature file
    :param num_features: feature count
    :return: extracted features
    """
    features_matrix = []
    with open(input_file, 'r') as file_stream:
        for matrix in file_stream:
            matrix_str = matrix.strip("\n").split(",")
            matrix_float = [float(matrix_str[i]) for i in range(len(matrix_str) - 1)]
            matrix_float = np.array(matrix_float)
            matrix_float = matrix_float.reshape(num_features, 35)
            features_matrix.append(matrix_float)
    return np.array(features_matrix)

def labels_from_file(input_file):
    labels = []
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip('\n')
            labels.append(line)
    return labels

def get_data_files(filepath, prefix, num_epochs, num_features=41,
                   model_type='lstm'):
    """
    model folder of type: type_prefix_features
    model file of type: prefix_features_epochs.model
    means and stddev file of type: means/stddev_prefix_numfeatures.npy
    """
    num_epochs = str(num_epochs)
    num_features = str(num_features)

    model_name = '_'.join([model_type, prefix, num_features])
    model_file = model_name + '_' + num_epochs + ".model"
    model_path = filepath + model_name + "/"
    means_file = '_'.join(["means", prefix, num_features]) + ".npy"
    stddevs_file = '_'.join(["stddev", prefix, num_features]) + ".npy"

    means_file = model_path + means_file
    stddevs_file = model_path + stddevs_file
    model_file = model_path + model_file

    return model_file, means_file, stddevs_file

def add_history(filepath, history_train, history_valid, metrics):
    """
    add to history from metrics collected on train, test data
    :param filepath:
    :param metrics: metrics to save to the file
    :param history_train: dict containing training metrics per epoch
    :param history_valid: tuple containig validation metrics per epoch
    """
    for i in range(len(metrics)):
        with open(filepath + "_" + metrics[i], "a+") as file:
            file.write(str(history_train[metrics[i]][0]))
            file.write(" ")
            file.write(str(history_valid[i]))
            file.write('\n')

def load_metric(filepath):
    """
    load the metric data from a file
    :param filepath: file to store metric data
    :return: np array containing metric data of type (train, validation)
    """
    history = list()
    with open(filepath, 'r') as file:
        for line in file:
            values = [np.float(i) for i in line.strip(" \n").split(" ")]
            values = np.asarray(values)
            history.append(values)
    return np.asarray(history)

def concat_files(dirpath, filenames, out_file, lines_per_file=-1):
    """
    concatenate multiple files into a single one
    :param dirpath: path to the files
    :param filenames: the list of filenames to concatenate
    :param out_file: where to store the concatenated data
    :param lines_per_file: how many lines to take from each file
    """
    if dirpath[-1] != '/':
        dirpath = dirpath + '/'
    out_path = dirpath + out_file
    if lines_per_file == -1:
        lines_per_file = 2 ** 20
    with open(out_path, 'w') as outfile:
        for filename in filenames:
            count = 0
            file_path = dirpath + filename
            with open(file_path) as infile:
                for line in infile:
                    if line != "" and count < lines_per_file:
                        outfile.write(line)
                        count += 1
                    elif count >= lines_per_file:
                        break

def smooth(x, window_len=11, window='hanning'):
    if window_len < 3:
        return x
    s = numpy.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[window_len:- window_len + 1]

def get_pitch_magnitude(audio_data_path, sample_rate):

    duration = librosa.get_duration(filename=str(audio_data_path))
    y, sr = librosa.load(audio_data_path, sr=sample_rate, duration=duration, mono=True)
    pitches, magnitudes = analyse(y, sr, fmin=80, fmax=250)

    return y, pitches, magnitudes

NUM_TIMESTAMPS = 35
BATCH_SIZE = 128
SAMPLE_RATE = 44100
MFCC_FEATURES = 13

def get_audio_from_intervals(audio_data, intervals):
    """
    concatenate audio data from the given intervals
    :param audio_data: data to be used
    :param intervals: intervals to keep from audio data
    :return: data containing only the kept intervals
    """
    new_audio = []
    for start, end in intervals:
        new_audio.extend(audio_data[start: end])
    return np.asarray(new_audio)

def shifted_delta_coefficients(mfcc_coef, d=1, p=3, k=7):
    """
    :param mfcc_coef: mfcc coefficients
    :param d: amount of shift for delta computation
    :param p: amount of shift for next frame whose deltas are to be computed
    :param k: no of frame whose deltas are to be stacked.
    :return: SDC coefficients
    reference code from: http://tiny.cc/q8d58y
    """
    total_frames = mfcc_coef.shape[1]
    padding = mfcc_coef[:p * (k - 1), :]

    mfcc_coef = np.hstack((mfcc_coef, padding)).T

    deltas = mfcc_to_delta(mfcc_coef, d).T
    sd_temp = []

    for i in range(k):
        temp = deltas[:, p * i + 1:]
        sd_temp.extend(temp[:, :total_frames])
    sd_temp = np.asarray(sd_temp)
    return sd_temp

def mfcc_to_delta(mfcc_coef, d):
    """
    compute delta coefficient
    :param mfcc_coef: mfcc coefficients where a row represents the feature
                      vector for a frame
    :param d: lag size for delta feature computation
    reference code from: http://tiny.cc/m6d58y
    """
    num_frames, num_coeff = mfcc_coef.shape
    vf = np.asarray([i for i in range(d, -1-d, -d)])
    vf = vf / sum(vf ** 2)
    ww = np.ones(d).astype(np.int)
    cx = np.vstack((mfcc_coef[ww, :], mfcc_coef))
    cx = np.vstack((cx, mfcc_coef[(num_frames * ww) - 1, :]))
    vx = np.reshape(lfilter(vf, 1, cx[:]), (num_frames + 2 * d, num_coeff))
    mask = np.ones(vx.shape, dtype=np.bool)
    mask[: d * 2, :] = False
    vx = np.reshape(vx[mask], (num_frames, num_coeff))
    return vx

def get_audio_features(path, extra_features=[""]):
    """
    currently supports extraction for the following features from a given audio file
    -  mfcc features
    -  delta mfcc features
    -  delta-delta mfcc features
    -  shifted delta coeff
    -  pitch estimate
    -  magnitude estimate
    example:
        features = get_audio_features("file_path.mp3", extra_features=["delta", "delta2"])
    :param extra_features: features to take into consideration
    :param path: filePath
    :return: mfcc_features: matrix with shape (num_features, NUM_TIMESTAMPS)
    """
    audio_time_series, pitch_vals, magnitude_vals = get_pitch_magnitude(path, SAMPLE_RATE)

    pitch_vals = smooth(pitch_vals, window_len=10)

    # process audio - remove silence
    intervals = librosa.effects.split(audio_time_series, top_db=18)
    audio_time_series = get_audio_from_intervals(audio_time_series, intervals)
    audio_time_series, _ = librosa.effects.trim(audio_time_series, top_db=10)

    mfcc_features = librosa.feature.mfcc(y=audio_time_series, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)

    if mfcc_features.shape[1] > NUM_TIMESTAMPS:
        mfcc_features = mfcc_features[:, 0:NUM_TIMESTAMPS]
    else:
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, NUM_TIMESTAMPS - mfcc_features.shape[1])),
                               mode='constant', constant_values=0)

    if pitch_vals.shape[0] > NUM_TIMESTAMPS:
        pitch_vals = pitch_vals[0:NUM_TIMESTAMPS]
    else:
        pitch_vals = np.pad(pitch_vals, ((0, 0), (0, NUM_TIMESTAMPS - pitch_vals.shape[0])),
                            mode='constant', constant_values=0)

    if magnitude_vals.shape[0] > NUM_TIMESTAMPS:
        magnitude_vals = magnitude_vals[0:NUM_TIMESTAMPS]
    else:
        magnitude_vals = np.pad(magnitude_vals, ((0, 0), (0, NUM_TIMESTAMPS - magnitude_vals.shape[0])),
                                mode='constant', constant_values=0)

    features = []
    features.extend(mfcc_features)

    if "delta" in extra_features:
        mfcc_delta = librosa.feature.delta(mfcc_features)
        features.extend(mfcc_delta)

    if "delta2" in extra_features:
        mfcc_delta_2 = librosa.feature.delta(mfcc_features, order=2)
        features.extend(mfcc_delta_2)

    if "sdc" in extra_features:
        sdc_coeff = shifted_delta_coefficients(mfcc_features)
        features.extend(sdc_coeff)

    if "pitch" in extra_features:
        features.append(pitch_vals)
        features.append(magnitude_vals)

    return np.asarray(features)
#########################################################################################################################

age_labels = {
    0: "fifties",
    1: "fourties",
    2: "sixties",
    3: "teens",
    4: "thirties",
    5: "twenties"
}

data_path = "audio/"
models_path = "model/"


def get_age(out_data):
    out_data = out_data[0]
    return age_labels[int(np.argmax(out_data))]


def main_program():

    age_weights, age_means, age_stddev = get_data_files(models_path, "age", 30)
    np.set_printoptions(precision=3)

    num_age_labels = len(age_labels)

    # declare the models
    age_model = lstm_age_model(num_age_labels)

    # load models
    age_model.load_weights(age_weights)

    mean_paths = [age_means]
    stddev_paths = [age_stddev]

    data_files = os.listdir(data_path)

    for data_file in data_files:
        data = get_audio_features(Path(data_path + data_file),
                                  extra_features=["delta", "delta2", "pitch"])
        data = np.array([data.T])

        data = norm_multiple(data, mean_paths, stddev_paths)

        age_predict = age_model.predict(data[0])

        age_print = "{} ==> AGE(lstm): {} age_prob: {}".format(data_file,get_age(age_predict).upper(), age_predict)
        #age_print = "{} ==> {}".format(data_file, get_age(age_predict))

        if get_age(age_predict)=="teens":
            #age_print = "young"
            #age_print = "{} ==> {}".format(data_file, get_age(age_predict))
            age_print = "{} ==> AGE(lstm): {}: {}".format(data_file,
                        "young", age_predict)
        elif get_age(age_predict)=="sixties":
            #age_print = "elders"
            #age_print = "{} ==> {}".format(data_file, get_age(age_predict))
            age_print = "{} ==> AGE(lstm): {}: {}".format(data_file,
                        "elders", age_predict)
        else:
            #age_print = "middle"
            #age_print = "{} ==> {}".format(data_file, get_age(age_predict))
            age_print = "{} ==> AGE(lstm): {}: {}".format(data_file,
                        "middle", age_predict)

        
        
        print(age_print)
        # print('=' * max(len(gender_print), len(age_print), len(lang_print)))


if __name__ == '__main__':
    main_program()



