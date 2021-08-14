# sys.path.insert(0, './scripts')
from logging import log
import scipy.io.wavfile as wav
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from json import load
from python_speech_features import mfcc, logfbank
import mlflow
from model_trainer import *
import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Activation, Bidirectional, TimeDistributed, Masking, Input, Dropout, GRU, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class ModelInference:
    def __init__(self,audio,sr,alphabets_path = 'data/alphabets_data.json',model_path = 'models/stacked-lstm_predict.h5'):
        self.alphabets_path = alphabets_path
        self.model_path = model_path
        self.FEAT_MASK_VALUE = 1e+10
        self.infer(audio,sr)

    def infer(self,audio,sr):
        self.load_files()
        self.prepare_feature(audio,sr)
        self.decode()

    def load_files(self):
        with open(self.alphabets_path, 'r', encoding='UTF-8') as alphabets_file:
            self.alphabets = load(alphabets_file)
        self.model_pred = tf.keras.models.load_model(self.model_path, custom_objects={'CTCLossLayer': CTCLossLayer})

    def prepare_feature(self,audio,sr):
        input_val = logfbank(
            audio, samplerate=sr, nfilt=26)
        input_val = (input_val - np.mean(input_val)) / \
            np.std(input_val)
        # transform in 3d array
        train_input = tf.ragged.constant(input_val, dtype=np.float32)
        train_input = tf.expand_dims(train_input, axis=0)
        self.train_seq_len = tf.cast(train_input.row_lengths(), tf.int32)
        self.train_input = train_input.to_tensor(
                default_value=self.FEAT_MASK_VALUE)
    
    def decode(self):
        decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(self.model_pred.predict(self.train_input), (1, 0, 2)), self.train_seq_len)
        d = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
        str_decoded = [''.join([self.alphabets['num_to_char'][str(x)]
                                for x in np.asarray(row) if x != -1]) for row in d]
        
        check_augmentation = False
        pred_statement = ""
        for index, prediction in enumerate(str_decoded):
            if(check_augmentation):
                prediction = prediction.replace(self.alphabets['num_to_char']['0'], ' ')
                pred_statement += prediction
            else:
                if(index % 7 == 0):
                    # Replacing space label to space
                    prediction = prediction.replace(self.alphabets['num_to_char']['0'], ' ')
                    pred_statement += prediction
        self.pred_statement = pred_statement

    def get_prediction(self):
        return self.pred_statement