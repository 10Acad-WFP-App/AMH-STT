import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = "../data/train/data.json"
SAVED_MODEL_PATH = "model.h5"
EPOCHS = 100
BATCH_SIZE = 100
PATIENCE = 5
LEARNING_RATE = 0.0001


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int32")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int32")

        input_length = input_length * \
            tf.ones(shape=(batch_len, 1), dtype="int32")
        label_length = label_length * \
            tf.ones(shape=(batch_len, 1), dtype="int32")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


# class CTCLoss(keras.losses.Loss):

#     def __init__(self, logits_time_major=False, blank_index=-1,
#                  reduction=keras.losses.Reduction.AUTO, name='ctc_loss'):
#         super().__init__(reduction=reduction, name=name)
#         self.logits_time_major = logits_time_major
#         self.blank_index = blank_index

#     def call(self, y_true, y_pred):
#         y_true = tf.cast(y_true, tf.int32)
#         y_true = tf.reshape(y_true,  [BATCH_SIZE, 1])
#         y_pred = tf.reshape(y_pred, [563, BATCH_SIZE, 1])
#         loss = tf.nn.ctc_loss(
#             labels=y_true,
#             logits=y_pred,
#             label_length=163,
#             logit_length=163)
#         return tf.reduce_mean(loss)

# model = Sequential()
# model.add(Bidirectional(LSTM(35, input_shape=X_train.shape, return_sequences=True)))
# # didn't add the hidden layers in this code snippet.
# model.add(Flatten())
# model.add(Dense((4480), activation='softmax'))

# model.compile(optimizer='adam',
#            loss=CTCLoss(),
#            metrics=['accuracy'])
#               loss=CTCLoss(),
#               metrics=['accuracy'])


def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r", encoding='UTF-8') as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    return X, y


def prepare_dataset(data_path, test_size=0.2, validation_size=0.2):
    """Creates train, validation and test sets.
    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """

    # load dataset
    X, y = load_data(data_path)

    # create train, validation, test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=validation_size)

    # add an axis to nd array
    # data = data.reshape((data.shape[0], data.shape[1], 1))
    X_train = X_train[..., np.newaxis]
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    # print(X_train.shape)
    # X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    # X_validation = X_validation.reshape(
    #     (X_validation.shape[0], X_validation.shape[1], 1))
    # print(X_validation.shape)

    # return X_train, y_train, X_validation, y_validation, X_test, y_test
    return X_train, y_train, X_validation, y_validation


def build_model(input_shape, loss=CTCLayer(), learning_rate=LEARNING_RATE):
    """Build neural network using keras.
    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param loss (str): Loss function to use
    :param learning_rate (float):
    :return model: TensorFlow model
    """

    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(300, (9, 9), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(300, (4, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(563, activation='relu'))
    # tf.keras.layers.Dropout(0.3)

    # Reshape
    model.add(tf.keras.layers.Reshape(target_shape=(563, 1)))

    # 1st Bidirectional layer
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        563, return_sequences=True, dropout=0.25)))
    model.add(tf.keras.layers.BatchNormalization())

    # 2nd Bidirectional layer
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        563, return_sequences=True, dropout=0.25)))
    model.add(tf.keras.layers.BatchNormalization())

    # 2 Dense Output
    model.add(tf.keras.layers.Dense(285, activation='relu'))
    model.add(tf.keras.layers.Dense(285, activation='relu'))

    # softmax output layer
    model.add(tf.keras.layers.Dense(285, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model


def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model
    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set
    :return history: Training history
    """

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()


def main():
    # generate train, validation and test sets
    # X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(
    #     DATA_PATH)
    X_train, y_train, X_validation, y_validation = prepare_dataset(
        DATA_PATH)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    X_train = np.asarray(X_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    X_validation = np.asarray(X_validation).astype('float32')
    y_validation = np.asarray(y_validation).astype('float32')

    # print(input_shape)

    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    # train network
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE,
                    X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # evaluate network on test set
    # test_loss, test_acc = model.evaluate(X_test, y_test)
    # print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
