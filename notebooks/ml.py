

import os
import sys
import tensorflow as tf
import scipy.io.wavfile as wav
import glob
import numpy as np
import six
from six.moves import xrange as range
import json
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
import mlflow
import math
import os
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense, Input,TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)
import librosa.display
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
from tensorflow.keras.callbacks import ModelCheckpoint 
sess = tf.compat.v1.Session(config=config)
import gc
 
import IPython.display as ipd
import soundfile

import random

import librosa

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# Constants 
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = 1
FEAT_MASK_VALUE = 1e+10

# Some configs
num_features = 13
num_units = 100
num_classes = 285 + 1 # 285(including space) + blamk label = 286

# Hyper-parameters
num_epochs = 25
num_layers = 1
batch_size = 2
initial_learning_rate = 0.005
momentum = 0.9

# Loading the data
file_path = glob.glob('../data/train/wav*/*.wav')
file_path=file_path[1:200]
audio_list = []
fs_list = []
dur_list = []
dropped_file_path = []

for file_name in file_path:
    audio,fs = librosa.load(file_name,sr=16000)
    dur = librosa.get_duration(audio,sr=16000)
    audio_list.append(audio)
    dur_list.append(dur)
    fs_list.append(fs)
        
# Create a dataset composed of data with variable lengths
inputs_list = []
for index in range(len(audio_list)):
    input_val = mfcc(audio_list[index], samplerate=fs_list[index])
    input_val = (input_val - np.mean(input_val)) / np.std(input_val)
    inputs_list.append(input_val)

# Transform in 3D Array
train_inputs = tf.ragged.constant([i for i in inputs_list], dtype=np.float32)
train_seq_len = tf.cast(train_inputs.row_lengths(), tf.int32)
train_inputs = train_inputs.to_tensor(default_value=FEAT_MASK_VALUE)
with open('../data/train1.json', 'r', encoding='UTF-8') as label_file:
    labels = json.load(label_file)
with open('../data/language_model.json', 'r', encoding='UTF-8') as language_file:
    alphabets = json.load(language_file)

# Reading Targets
original_list = []
targets_list = []

for path in file_path:
    file_name = path[:-4].split('wav')[1][1:]
    # Read Label
    label = labels[file_name]
    original = " ".join(label.strip().split(' '))
    original_list.append(original)
    # print(original)
    target = original.replace(' ', '  ')
    # print('step-1. ',target)
    target = target.split(' ')
    # print('step-2. ', target)
    # Adding blank label
    target = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in target])
    # print('step-3. ', target)
    # Transform char into index
    target = np.asarray([alphabets['char_to_num'][x] for x in target])
    # print('step-4. ', target)
    targets_list.append(target)
# Creating sparse representation to feed the placeholder
train_targets = tf.ragged.constant([i for i in targets_list], dtype=np.int32)
train_targets_len = tf.cast(train_targets.row_lengths(), tf.int32)
train_targets = train_targets.to_sparse()
val_inputs, val_targets, val_seq_len, val_targets_len = train_inputs, train_targets, train_seq_len, train_targets_len



class CTCLossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        labels = inputs[0]
        logits = inputs[1]
        label_len = inputs[2]
        logit_len = inputs[3]

        logits_trans = tf.transpose(logits, (1,0,2))
        label_len = tf.reshape(label_len, (-1,))
        logit_len = tf.reshape(logit_len, (-1,))
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits_trans, label_len, logit_len, blank_index=-1))
        # define loss here instead of in compile
        self.add_loss(loss)

        # Decode
        decoded, _ = tf.nn.ctc_greedy_decoder(logits_trans, logit_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),labels))
        self.add_metric(ler, name='ler', aggregation='mean')

        return logits


# Defining Training Cells
cells = []
for _ in range(num_layers):
    cell = tf.keras.layers.LSTMCell(num_units)
    cells.append(cell)

stack = tf.keras.layers.StackedRNNCells(cells)
# Definning Input Parameters
input_feature = tf.keras.layers.Input((None, num_features), name='input_feature')
input_label = tf.keras.layers.Input((None,), dtype=tf.int32, sparse=True, name='input_label')
input_feature_len = tf.keras.layers.Input((1,), dtype=tf.int32, name='input_feature_len')
input_label_len =tf.keras.layers.Input((1,), dtype=tf.int32, name='input_label_len')

layer_masking = tf.keras.layers.Masking(FEAT_MASK_VALUE)(input_feature)
layer_rnn = tf.keras.layers.RNN(stack, return_sequences=True)(layer_masking)
# layer_drop = tf.keras.layers.Dropout(0.2, seed=42)(layer_rnn)
layer_output = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.0,0.1), bias_initializer='zeros', name='logit')(layer_rnn)

layer_loss = CTCLossLayer()([input_label, layer_output, input_label_len, input_feature_len])
# Create models for training and prediction
model_train = tf.keras.models.Model(inputs=[input_feature, input_label, input_feature_len, input_label_len],
            outputs=layer_loss)

model_predict = tf.keras.models.Model(inputs=input_feature, outputs=layer_output)
# Compile Training Model with selected optimizer
optimizer = tf.keras.optimizers.Adam(initial_learning_rate, momentum)
model_train.compile(optimizer=optimizer)
checkpointer = ModelCheckpoint(filepath='../models/'+"RNN"+'.h5',monitor='val_loss',verbose=1, save_best_only=True, mode='min')
# Training, Our y is already defined so no need
try:
    experiment_id = mlflow.create_experiment("Stacked RNN(LSTM): 50 Cells")
    experiment = mlflow.get_experiment(experiment_id)
except mlflow.exceptions.MlflowException:
    experiment = mlflow.get_experiment_by_name("Stacked RNN(LSTM): 50 Cells")

mlflow.tensorflow.autolog()
history=model_train.fit(x=[train_inputs, train_targets, train_seq_len, train_targets_len], y=None,validation_data=([val_inputs, val_targets, val_seq_len, val_targets_len], None),batch_size=batch_size, callbacks=[checkpointer],epochs=num_epochs)
