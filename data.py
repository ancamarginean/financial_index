import pandas as pd
import numpy as np

def read_data():
    #train_file_path = "/content/gdrive/My Drive/oradea/date.csv"
    train_file_path = "/mnt/7ac346ab-63e9-4291-b1d0-6ac9ad09954a/work/financial/date.csv"
    desc = pd.read_csv(train_file_path)
    data = desc.to_numpy()
    return data


def build_data_for_one_year_based(data):
    (m, n) = data.shape
    j = 0
    proc_data = []
    y = []
    t = 0.0015
    for i in range(m):
        if data[i, 1] < 2018 and not np.isnan(data[i, 10]) and not np.isnan(data[i + 1, 10]):
            line = data[i, 2:11].astype(np.double)  # include index for current year
            line = np.append(line, [i])  # index of line
            proc_data.append(line)
            if data[i + 1, 10] > data[i, 10]:
                increase = 1
            else:
                increase = 0
            if data[i + 1, 10] > data[i, 10] + t:
                inc1 = 2  # increased above t
            else:
                if data[i + 1, 10] < data[i, 10] - t:
                    inc1 = 1
                else:
                    inc1 = 0  # similar  to increase/decrease, but with a certain threshold
            y.append([100 * float(data[i + 1, 10]), increase, inc1])
    y = np.array(y)
    print(np.unique(y[:, 1], return_counts=True))  # distribution of classes increase/decrease
    print(np.unique(y[:, 2], return_counts=True))
    return (proc_data, y)


def build_data_for_two_years_based(data):
    (m, n) = data.shape
    j = 0
    line0 = data[0, 2:11].astype(np.double)
    proc_data = []
    y = []
    t = 0.0015
    for i in range(0, m, 1):
        if data[i, 1] < 2017 and not np.isnan(data[i, 10]) and not np.isnan(data[i + 1, 10]) and not np.isnan(
                data[i + 2, 10]):
            line0 = data[i, 2:11].astype(np.double)  # include index for current year
            line1 = data[i + 1, 2:11].astype(np.double)  # include index for current year
            line = np.append(line0, line1)
            line = np.append(line, [i])  # index of line
            proc_data.append(line)
            line0 = line1  # get ready for the next year
            if data[i + 2, 10] > data[i + 1, 10]:
                increase = 1
            else:
                increase = 0
            if data[i + 2, 10] > data[i + 1, 10] + t:
                inc1 = 2  # increased above t
            else:
                if data[i + 2, 10] < data[i + 1, 10] - t:
                    inc1 = 1
                else:
                    inc1 = 0  # similar  to increase/decrease, but with a certain threshold
            y.append([100 * float(data[i + 1, 10]), increase, inc1])
    y = np.array(y)
    print(np.unique(y[:, 1], return_counts=True))  # distribution of classes increase/decrease
    print(np.unique(y[:, 2], return_counts=True))
    return (proc_data, y)


def prepare_data(oneyear, data):
    if oneyear:
        return build_data_for_one_year_based(data)
    else:
        return build_data_for_two_years_based(data)

def build_sequences(data):
    no_of_steps = 7
    no_of_features = 9
    sequences = np.ones((56, no_of_steps, no_of_features))
    y2s = np.ones((56, no_of_steps, 1))  # maybe try to add the other values too!!!!!
    # transform to n companies * time_steps*features (multivariate).. single step
    companies_number = 56
    k = 0
    for i in range(companies_number):
        sequence = np.zeros((no_of_steps, no_of_features))
        y2 = np.zeros((no_of_steps, 1))
        complete = 1
        for j in range(no_of_steps):  # the last year 2018 is not considered for seq, therefore test
            if not np.isnan(data[i * 8 + j, 10]) and not np.isnan(data[i * 8 + j + 1, 10]):
                sequence[j] = data[i * 8 + j, 2:11]
                y2[j] = data[i * 8 + j + 1, 10]  # next year
            else:
                complete = 0
        if complete:
            sequences[k] = sequence
            y2s[k] = y2
            k = k + 1
    # create from the beginning seq2seq, and at the first experiment take only last
    cleaned_sequences = [x for x in sequences if len(np.unique(x)) > 1]
    y2s = y2s[:48]
    return cleaned_sequences, y2s