import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint


def checkpoint(classification, seq = False):
  if classification:
    checkpoint_name = 'models/Classification_Weights-{epoch:03d}--{val_accuracy:.5f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_accuracy',  save_best_only = True, mode ='auto') # verbose = 1,
  else :
    if seq:
      checkpoint_name = 'models/RegressionSeq_Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    else:
      checkpoint_name = 'models/Regression_Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss',  save_best_only = True, mode ='auto')

  return checkpoint



def build_model(features_no): #for regression on window based data: small network due to the size of the dataset
  model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[features_no]),
    #layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # to be added kernel_initializer='normal',activation='linear'
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


def build_model_classification(features_no): #for classification
  model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[features_no]),
    layers.Dropout(0.4),
   # layers.Dense(32, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.4),
    #layers.Dense(1, activation='sigmoid')
    layers.Dense(3, activation='softmax')
  ])

  optimizer = tf.keras.optimizers.Adam()


  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

def build_simple_rnn_seq2vector():
  no_of_steps = 7
  no_of_features = 9
  model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(24, input_shape=(no_of_steps, no_of_features), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.SimpleRNN(12),
    # tf.keras.layers.LSTM(12 ),
    tf.keras.layers.Dense(1)
  ])
  print(model.summary())
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  return model

def build_lstm_seq2seq():
  no_of_steps = 7
  no_of_features = 9
  seq2seq_lstm_model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(24, input_shape=(no_of_steps, no_of_features), return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(12, return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      #tf.keras.layers.LSTM(12 ),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
  ])

  seq2seq_lstm_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  print(seq2seq_lstm_model.summary())
  return seq2seq_lstm_model