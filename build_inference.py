from data import prepare_data, read_data, build_sequences
from model import build_model, build_model_classification, checkpoint, build_simple_rnn_seq2vector, build_lstm_seq2seq
from preprocess import split, split_transform, only_index_2018, only_index_all_years, only_index_all_years_multiplied_100
#from utils import plot_history_regression, plot_history_classification
import datetime
import tensorflow as tf
import numpy as np




def get_data(type, oneyear): #without sequences
    data=read_data()
    proc_data, y = prepare_data(oneyear, data)
    x_train, x_test, y_train, y_test = split(proc_data, y, type)
    return x_train, x_test, y_train, y_test


def train_regression():
    oneyear=True # two years care considered
    type = 0
    x_train, x_test, y_train, y_test = get_data(type, oneyear)
    if oneyear:
        model = build_model(9)
    else:
        model = build_model(18)
    model.summary()

    log_dir = "logs/fit_regression/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(x_train[:, :-1], y_train[:, 0], epochs=90, batch_size=20, verbose=2,
                        validation_data=(x_test[:, :-1], y_test[:, 0]),
                        callbacks=[checkpoint(False), tensorboard_callback])

def train_classification():
    oneyear = False  # two years are considered
    type = 2   # 1 and 2 - classification with/without threshold for increase/decrease

    x_train, x_test, y_train, y_test = get_data(type, oneyear)
    if oneyear:
        model = build_model_classification(9)
    else:
        model = build_model_classification(18)

    model.summary()
    log_dir = "logs/fit_classification/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #the value of type determines which interpretation of the index is considered
    history_classification = model.fit(x_train[:, :-1], y_train[:, type], epochs=250, verbose=2,
                                                      validation_data=(x_test[:, :-1], y_test[:, type]),
                                                      callbacks=[checkpoint(True), tensorboard_callback])
    #example of inference:
    # predicts_test = model.predict(x_test[:, :-1])

#check whether the correct data are included in teh sequences
def get_company(first_year_first_two_values, data):
  company_name = [x[0] for x in data if np.array_equal(x[2:4],first_year_first_two_values)]
  return (company_name,[x[10] for x in data if x[0] in company_name])


def train_regression_sequences():
    data = read_data()
    cleaned_sequences, y2s = build_sequences(data)
    seq2vec = False #False for seq2seq - all years are included in y, seq2vec = only the last one

    # x_train, x_test, y_train, y_test = split_transform(only_index_2018)
    x_train, x_test, y_train, y_test = split_transform(only_index_all_years, cleaned_sequences, y2s)
    # x_train, x_test, y_train, y_test = split_transform(only_index_all_years_multiplied_100)

    EPOCHS = 50
    log_dir = "logs/fit_rnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if seq2vec:
        model = build_simple_rnn_seq2vector()
        history_rnn = model.fit(x_train, y_train[:, -1], epochs=EPOCHS + 10,
                                validation_data=(x_test, y_test[:, -1]),
                                callbacks=[checkpoint(False, seq=True), tensorboard_callback])
    else:
        model = build_lstm_seq2seq()
        history_seq2seq_rnn = model.fit(x_train, y_train, epochs=80,
                                                     validation_data=(x_test, y_test), verbose=1, shuffle=True,
                                                     callbacks=[checkpoint(False, seq=True), tensorboard_callback])


#change parameters inside the function in order to reproduce the experiments

#Uncomment one
#train_regression()
#train_classification()
#train_regression_sequences()

