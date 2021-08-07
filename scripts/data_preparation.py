import librosa
import os
import json
import glob

DATASET_PATH = "../data/train"
JSON_PATH = "../data/train/data.json"
SAMPLES_TO_CONSIDER = 288000  # 1 sec. of audio
label_length = 163


def interpret(dictionary, value):
    if(len(value) < label_length):
        difference = int(label_length - len(value) / 2)
        returnList = ['<space>'] * difference
        endList = ['<space>'] * (label_length - len(value) - difference)
    else:
        returnList = []
        endList = []
    for i in value:
        if(i == " "):
            returnList.append(dictionary['<space>'])
        else:
            returnList.append(dictionary[i])

    return returnList.extend(endList)


def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        # actual label
        "mapping": [],
        # labels interpreted to numbers
        "labels": [],
        # mfcc list transposed
        "MFCCs": [],
        # path to each audio file
        "files": [],
        # char_to_num
        'char_to_num': None,
        # num_to_char
        'num_to_char': None,
        # alphabet_size
        'alphabet_size': None
    }

    with open('../data/labels.json', 'r', encoding='UTF-8') as label_file:
        labels = json.load(label_file)

    with open('../data/language_model.json', 'r', encoding='UTF-8') as lang_file:
        language = json.load(lang_file)
        data['char_to_num'] = language['char_to_num']
        data['alphabet_size'] = language['alphabet_size']
        data['num_to_char'] = language['num_to_char']
    # loop through all sub-dirs
    # for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    for file_path in glob.glob('../data/train/wav2/*.wav'):
        # ensure we're at sub-folder level
        # if dirpath is not dataset_path:
        try:
            name = file_path[:-4].split('wav2')[1][1:]
            # save label (i.e., sub-folder name) in the mapping
            # label = dirpath.split("/")[-1]
            label = labels[name]
            data["mapping"].append(label)
            # print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            # for f in filenames:
            #     file_path = os.path.join(dirpath, f)

            # load audio file and slice it to ensure length consistency among different files
            signal, sample_rate = librosa.load(file_path, sr=16000)

            # drop audio files with less than pre-decided number of samples
            if len(signal) >= SAMPLES_TO_CONSIDER:

                # ensure consistency of the length of the signal
                signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)

            # store data for analysed track
            data["MFCCs"].append(MFCCs.T.tolist())
            data["labels"].append(interpret(data['char_to_num'], label))
            data["files"].append(file_path)
            # print("{}: Complete".format(file_path))

        except Exception as e:
            print(e)
            pass

    # save data in json file
    with open(json_path, "w", encoding='UTF-8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
