#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:42:31 2020

@author: sollyboukman
"""

import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt

'''
Reading Size represents ammount of chartevents features to check per patient
This is an arbitrary value set by the user.
Data has been dificult to work with because patients may have SIGNIFICANTLY more or less
readings within the same time frame. (Some patients may have 5 rows in 10 hours while others have 50 rows)
'''
reading_size = 480 #Average 7 readings per row * 100 rows

class Model(tf.keras.Model):
    '''
    LSTM baseline model
    *Currently only supports IHM prediction*
    '''
    def __init__(self, num_timesteps, model_type):

        super(Model, self).__init__()

        self.model_type = model_type
        self.batch_size = 10 #VARIABLE
        self.num_timesteps = num_timesteps
        self.lstm_size = 120
        self.learning_rate = 0.1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.loss_list = []
        self.accuracy_list = []
        self.auprc_list = []

        self.W1 = tf.keras.layers.Dense(2, activation = 'softmax')
        self.Rnn = tf.keras.layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)

    def call(self, inputs):
        '''
        LSTM Layer output: (batch_size, reading_size, self.lstm_size)
        Dense Layer output: (batch_size, reading_size, 2)
        Droupout of .2
        Each reading within reading_size has an expected probability for 0 or 1
        '''

        batch_element, last_output, cell_state = self.Rnn(inputs)
        dense = self.W1(batch_element)
        dense_1 = tf.nn.dropout(dense, .2)

        return dense_1 #, last_output

    def probs_labels(self, probs):
        '''
        Collapese (batch_size, reading_size, 2) to (batch_size, 2)
        by averaging down (reading_size, 2) for each batch_size
        '''
        probs_last_row_label = []
        store_predict_within = []

        for row in probs:

            for i in row:
                predict = np.argmax(i)
                store_predict_within.append(predict)

            average_predict = sum(store_predict_within) / len(store_predict_within)
            probs_last_row_label.append([1-average_predict, average_predict])
            store_predict_within = []


        return tf.cast(probs_last_row_label, tf.float32)

    def expand_labels(self, labels):
        '''
        Takes actual labels (batch_size,) and expands them to (batch_size, reading_size, 2)
        by applying each label down for reading_size
        '''
        output = []
        inner_array = []

        for row in labels:
            if row == 0:
                one_hot = [1, 0]
            else:
                one_hot = [0, 1]
            inner_array = np.repeat([one_hot], reading_size, axis=0)
            output.append(inner_array)

        return tf.cast(output, tf.float32)


    def loss(self, probs, labels):
        '''
        Current loss layer:
        probs = (batch_size, reading_size, 2)
        labels = (batch_size, reading_size, 2)
        When using Keras layer:
        probs = (batch_size, reading_size, 2)
        probs_labels = (batch_size, 2)
        labels = (batch_size, ) -> transform this to (batch_size, 2)
        '''

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, probs))

    def accuracy(self, logits, labels):
        '''
        Collapse logits (batch size, reading_size, 2) and labels (batch size, reading_size, 2)
        to both be (batch_size, 2) and compare labels
        '''
        logits = self.probs_labels(logits)
        labels = self.probs_labels(labels)
        #print(logits)
        #print(labels)
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def auprc(self, logits, labels):
        '''
        Collapse logits (batch size, reading_size, 2) and labels (batch size, reading_size, 2)
        to both be (batch_size, 2) and compare labels
        AUPRC positive label set to 0, so we purposefully index for the expected probability for 0 in logits
        '''
        logits = self.probs_labels(logits).numpy()
        labels = self.probs_labels(labels).numpy()
        labels = np.argmax(labels, axis=1)
        new_logits = []

        for row in logits: #Since we set pos_label = 0, we want to ONLY prob for 0 labels in logits
            new_logits.append(row[0])

        logits = np.array(new_logits)
        sklearn_out = sklearn.metrics.average_precision_score(labels, logits, pos_label=0)

        return tf.cast(sklearn_out, tf.float32)


def train_ihm(model, time_series, labels):
    '''
    In-hosipital-mortality training
    '''
    size_inputs = len(labels)
    num_batches = size_inputs // model.batch_size

    #range_tensor = tf.range(0, size_inputs)
    #shuffled_range = tf.random.shuffle(range_tensor)
    #train_inputs = tf.gather(time_series, shuffled_range)
    #train_labels = tf.gather(labels, shuffled_range)
    train_inputs = np.expand_dims(time_series, -1)
    train_labels = [float(i) for i in labels]
    accuracy_list = []

    for i in range(num_batches):
        starting_batch_index = i * model.batch_size
        next_batch_index = starting_batch_index + model.batch_size

        inputs_batch = tf.convert_to_tensor(train_inputs[starting_batch_index:next_batch_index])
        labels_batch = tf.convert_to_tensor(train_labels[starting_batch_index:next_batch_index])


        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            logits = model.call(inputs_batch)
            labels_batch = model.expand_labels(labels_batch)
            loss = model.loss(logits, labels_batch)
            print("Loss at batch " + str(i) + ": " + str(loss))


        model.loss_list.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return model.loss_list

def test_ihm(model, time_series, labels):
    '''
    In-hosipital-mortality testing
    '''
    size_inputs = len(labels)
    num_batches = size_inputs // model.batch_size
    accuracy_list = []
    auprc_list = []

    #range_tensor = tf.range(0, size_inputs)
    #shuffled_range = tf.random.shuffle(range_tensor)
    #train_inputs = tf.gather(time_series, shuffled_range)
    #train_labels = tf.gather(labels, shuffled_range)
    train_inputs = np.expand_dims(time_series, -1)
    train_labels = [float(i) for i in labels]

    for i in range(num_batches):
        starting_batch_index = i * model.batch_size
        next_batch_index = starting_batch_index + model.batch_size

        inputs_batch = tf.convert_to_tensor(train_inputs[starting_batch_index:next_batch_index])
        labels_batch = tf.convert_to_tensor(train_labels[starting_batch_index:next_batch_index])

        logits = model.call(inputs_batch)
        labels_batch = model.expand_labels(labels_batch)
        loss = model.loss(logits, labels_batch)
        accuracy = model.accuracy(logits, labels_batch)
        auprc = model.auprc(logits, labels_batch)

        auprc_list.append(auprc)
        accuracy_list.append(accuracy)

    model.accuracy_list = accuracy_list
    model.auprc_list = auprc_list

    return np.mean(accuracy_list), np.mean(auprc_list)

def create_time_series():
    '''
    Returns an array with each row representing a hadm_id chartevents concatenated
    *Note we used subject_id previously but switched to hadm_id for full model*
    '''
    chartevents = "chartevents.csv"
    arr = np.genfromtxt(chartevents, dtype=np.dtype(str), delimiter=',', invalid_raise = False, max_rows = 500000) #max_rows=1500000
    arr = np.delete(arr, (0), axis=0) #delete first row
    store_dict = {}

    for row in arr:
        if len(row) != 15: #skip over rows with 11 columns bug
            continue
        row = np.delete(row, [5, 6, 10]) #remove ROW_ID, CHARTIME, STORETIME, VALUEUOM
        empty_string_index = np.argwhere(row == '') #remove empty strings
        row = np.delete(row, empty_string_index)
        hadm_id = row[2]
        if hadm_id in store_dict:
            current_time_series = store_dict.get(hadm_id)
            #store_dict[subject_id] = np.append(current_time_series, row)
            store_dict[hadm_id] = np.append(current_time_series, [float(i) for i in row])
        else:
            #store_dict[subject_id] = row
            store_dict[hadm_id] = [float(i) for i in row] #turn to float

    values = store_dict.values()
    time_series = list(values)

    return time_series, store_dict

def get_label(hadm_id, mortality_array):
    '''
    Returns mortality of hadm_id
    *Assume hadm_id is in mortality_array*
    '''
    for row in mortality_array:
        if row[2] == hadm_id:
            return row[1]


def create_labels(time_series):
    '''
    Returns an array with each index representing mortality of each hadm_id from time series.
    Returns an updated time_series array if the subject_id can not be found
    In this step we remove [ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID] as well
    '''
    ytrain_path = r"y_train.csv"
    ytest_path = r"y_test.csv"
    icu_mortality_path = r"/Users/sollyboukman/Desktop/icu_mortality.csv"
    ytrain = np.genfromtxt(ytrain_path, dtype=np.dtype(str), delimiter=',')
    
    ytest = np.genfromtxt(ytest_path, dtype=np.dtype(str), delimiter=',')
    icu_mortality = np.genfromtxt(icu_mortality_path, dtype=np.dtype(str), delimiter=',')
    #print(icu_mortality)
    
    label_array = []
    remove_list = []
    
    #print(remove_list)

    for row in time_series:
        count = 0
        hadm_id = str(int(row[2]))
        #print("hadm id", hadm_id)

        values_to_delete = [row[0], row[1], row[2], row[3]] #remove [ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID]
        for value in values_to_delete:
            row = np.delete(row, np.where(row == value))
        time_series[count] = row
        #print("here1")
        if any(hadm_id in sublist for sublist in icu_mortality): #subject_id is in icu_mortality
            label = get_label(hadm_id, icu_mortality)
            #print(label)
            label_array.append(label)
            count += 1
        
        else: #Delete row from time_series,
            remove_list.append(count) #append index of row to remove
            count += 1
            
        #print('here2')

    #time_series = [i for i in time_series if i not in remove_list] #Remove from time_Series contents of remove_list
    for row in remove_list:
        time_series.pop(row)

    return label_array, time_series

def filter_time_series(time_series):
    '''
    Returns a time_series with consistent row size
    Limits each patient (a row) to have size of reading_size or remove patient that does not have that size
    *Check comment under reading_size for explanation*
    '''
    count = 0
    filtered_series = []
    #[x for x in time_series if len(x) >= readings_limit] #Remove all times below readings_limit

    for row in time_series:

        if len(row) >= reading_size:
            filtered_series.append(row)

    for row in filtered_series:
        filtered_series[count] = row[:reading_size] #Set all items up to reading_limit
        count += 1
        
    

    return filtered_series

def plot_value(list, xlabel, ylabel, title):
    '''
    Takes in a list of loss / accuracy / auprc and corresponding labels
    to output a matplotlib PLT
    '''
    plot_list = np.array(list)
    plt.plot(plot_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()



def main():
    '''
    Create testing time series, testing labels, training time series, training labels
    Model limited to In-Hospital-Mortality, num_timesteps = first 48 hours
    prints accuracy of testing data
    '''

    initial_time_series, HADM_DICT = create_time_series() #HADM_DICT IS WHAT U NEED SOL
    print(initial_time_series)
    initial_time_series = filter_time_series(initial_time_series)
    labels_full, time_series_full = create_labels(initial_time_series)
    num_subjects = len(time_series_full)
    num_test = int(.2 * num_subjects) #10% test data
    num_train = num_subjects - num_test

    time_series_full = np.reshape(time_series_full, (num_subjects, reading_size))
    labels_full =  np.reshape(labels_full, (num_subjects, ))

    time_series, time_series_test = time_series_full[:num_train, :], time_series_full[num_train:, :]
    labels, labels_test = labels_full[:num_train], labels_full[num_train:]


    model_type = "IHM"
    num_timesteps = 48
    model = Model(num_timesteps, model_type)
    loss_list = train_ihm(model, time_series, labels)
    accuracy, auprc = test_ihm(model, time_series_test, labels_test)

    list_label_test = list([int(i) for i in labels_test])
    num_zeros_test = list_label_test.count(0)
    num_ones_test = list_label_test.count(1)

    print("Ratio of 0s to 1s in testing set: " + str(num_zeros_test / (num_zeros_test + num_ones_test)))
    print("accuracy is: " + str(accuracy))
    print("auprc is: " + str(auprc))

    plot_value(loss_list, "Number of Batches", "Loss", "Loss over Time")
    plot_value(model.accuracy_list, "Number of Batches", "Accuracy", "Accuracy over Time")
    plot_value(model.auprc_list, "Number of Batches", "AUPRC", "AUPRC over Time")


if __name__ == '__main__':
    main()
