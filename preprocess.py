from sklearn.model_selection import train_test_split
import numpy as np

def split(proc_data, y, type):
  x_train, x_test, y_train, y_test = train_test_split(proc_data, y, test_size=0.25) #, random_state=42) #for classification
  x_train=np.array(x_train)
  y_train=np.array(y_train)
  x_test=np.array(x_test)
  y_test=np.array(y_test)
  return x_train, x_test, y_train, y_test

#forsequences
def split_transform(fy, sequences, y2s):
  x_train, x_test, y_train, y_test = train_test_split(sequences, fy(y2s), test_size=0.2) #only the last from y
  x_train=np.array(x_train)
  x_train[:,:,-1]=fy(x_train[:,:,-1])
  y_train=np.array(y_train)
  x_test=np.array(x_test)
  x_test[:,:,-1]=fy(x_test[:,:,-1])
  y_test=np.array(y_test)

  return (x_train, x_test, y_train,y_test)

def only_index_2018(y):
  return np.multiply(np.sign(y[:,-1]),np.power(np.abs(y[:,-1]),0.5)) #y[:, -1]

def only_index_all_years(y):
  return np.multiply(np.sign(y[:,:]),np.power(np.abs(y[:,:]),0.2))  #y

def only_index_all_years_multiplied_100(y):
  return y*100