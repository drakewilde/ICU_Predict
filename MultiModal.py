
import tensorflow as tf
import numpy as np
import pandas as pd
import time_series as ts
import baseline_bert as bert
import sklearn.metrics
import random


class MultiModal(tf.keras.Model):
    '''
    MultiModal model containing a LSTM to process time series data and a transformer to process note data
    '''
    
    def __init__(self, batchSz, TStrain, TStest, BertTrain, BertTest, TrainLabels, TestLabels, LSTMtrained, hidden_size):
        print('here1')
        super(MultiModal, self).__init__()
        print('here2')
        self.batchSz = batchSz
        self.TStrain = TStrain
        self.TStest = TStest
        self.BertTrain = BertTrain
        self.BertTest = BertTest
        self.FF2 = tf.keras.layers.Dense(2, activation='softmax')
        self.FF1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.LSTM = LSTMtrained
        self.Tformer = bert.Baseline(2e-5, 1, BertTrain, BertTest, TrainLabels, TestLabels, self.batchSz)
        self.TrainLabels = TrainLabels
        self.TestLabels = TestLabels
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
        if len(self.BertTest) != len(self.TStest):
            self.BertTest = self.BertTest[:len(self.TStest)]
        
    def forwardPass(self, TSinputs, Binputs):
        '''
        Expects batch size inputs for time series data and note data
        calls a forward pass for the multimodal model

        '''
        
        MMinput = tf.concat([self.LSTM.probs_labels(TSinputs), Binputs], axis=1)
        
        FF1out = self.FF1(MMinput)
        
        FF2out = self.FF2(FF1out)
        
        return FF2out
    
    def train(self):
        '''
        Trains the multimodal model
        '''
        
        lossList = []
        
        for i in range(0, len(self.TStrain), self.batchSz):
            
            inputLabels = self.TrainLabels[i:i+self.batchSz]
            
            BertinputBatch = self.BertTrain[i:i+self.batchSz]
            
            bertOut = self.Tformer.forwardPass(BertinputBatch)
            
            TSinputBatch = self.TStrain[i:(i+self.batchSz),:]
            
            TSout = self.LSTM.call(TSinputBatch)
            
            TSout = self.LSTM.probs_labels(TSout)
            
            
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                logits = self.forwardPass(TSout, bertOut[0])
                
                #print(logits)
                
                loss = self.loss(logits, inputLabels)
                
            
            lossList.append(loss)
            
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            
        return lossList
            
            
    
    def test(self):
        '''
        Tests the multimodal model

        '''
        
        accuracy_list = []
        auprc_list = []
        
    
        for i in range(0, len(self.TStest), self.batchSz):
            
            inputLabels = self.TestLabels[i:i+self.batchSz]
            
            TSinputBatch = self.TStest[i:(i+self.batchSz),:]
            
            BertinputBatch = self.BertTest[i:i+self.batchSz]
            
            
            bertOut = self.Tformer.forwardPass(BertinputBatch)
            
            TSout = self.LSTM.call(TSinputBatch)
            
            TSout = self.LSTM.probs_labels(TSout)
            
            logits = self.forwardPass(TSout, bertOut[0])
            
            batchAcc = self.accuracy(logits, inputLabels)
            
            batchAUPRC = self.auprc(logits, inputLabels)
            
            accuracy_list.append(batchAcc)
            
            auprc_list.append(batchAUPRC)
            
            
        return accuracy_list, auprc_list
    
    
    def loss(self, probs, labels):
        '''
        Calculates the loss for a batch of inputs 

        '''
        
        labels = [int(label) for label in labels]
     
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, 2), probs))

    
    def accuracy(self, logits, labels):
        '''
        Calculates accuracy for a batch of inputs
        '''
        labels = [int(label) for label in labels]

        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(labels, 2), 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def auprc(self, logits, labels):
        '''
        Calculates AUPRC for a batch of inputs
        '''
        labels = [int(label) for label in labels]
        
        labels = np.argmax(tf.one_hot(labels, 2), axis=1)
        new_logits = []

        for row in logits: #Since we set pos_label = 0, we want to ONLY prob for 0 labels in logits
            new_logits.append(row[0])

        logits = np.array(new_logits)
        print("auprc logits")
        print(logits)
        print("auprc labels")
        print(labels)

        sklearn_out = sklearn.metrics.average_precision_score(labels, logits, pos_label=0)

        return tf.cast(sklearn_out, tf.float32)

def main():
    '''
    Create testing time series, testing labels, training time series, training labels
    Model limited to In-Hospital-Mortality, num_timesteps = first 48 hours
    prints accuracy of testing data
    
    print("Getting Timeseries Data")
    initial_time_series, HADM_DICT = ts.create_time_series() #HADM_DICT IS WHAT U NEED SOL
    print('calling filter time series')
    initial_time_series = ts.filter_time_series(initial_time_series)
    labels_full, time_series_full = ts.create_labels(initial_time_series)
    num_subjects = len(time_series_full)
    num_test = int(.2 * num_subjects) #10% test data
    num_train = num_subjects - num_test
    
    time_series_full = np.reshape(time_series_full, (num_subjects, ts.reading_size))
    labels_full =  np.reshape(labels_full, (num_subjects, ))
    
    time_series, time_series_test = time_series_full[:num_train, :], time_series_full[num_train:, :]
    labels, labels_test = labels_full[:num_train], labels_full[num_train:]
    
    print('Getting Note Data')
    p = bert.PreprocessNote("clinical_notes.csv")
    
    y = p.readNotes()
    
    p.populateStayTime(y)
    
    
    p.pupulateLastNote(y)
    
    p.cleanNotes()
    
    all_notes = []
    
    every_note = list(p.stayLastNote.values())
    
    for id in list(HADM_DICT.keys()):
        if id in p.stayLastNote:
            all_notes.append(p.stayLastNote[id])
        else:
            randidx = random.randint(0, len(every_note))
            
            all_notes.append(every_note[randidx])
            
    
    notesTrain, notesTest = all_notes[:num_train], all_notes[num_train:]
    
    model_type = "IHM"
    num_timesteps = 48
    model = ts.Model(num_timesteps, model_type)
    print('Running Time Series Model')
    loss_list = ts.train_ihm(model, time_series, labels)
    accuracy, auprc = ts.test_ihm(model, time_series_test, labels_test)
    
    print("accuracy is: " + str(accuracy))
    print("auprc is: " + str(auprc))
    
    ts.plot_value(loss_list, "Number of Batches", "Loss", "Loss over Time")
    ts.plot_value(model.accuracy_list, "Number of Batches", "Accuracy", "Accuracy over Time")
    ts.plot_value(model.auprc_list, "Number of Batches", "AUPRC", "AUPRC over Time")
    '''
    '''
    print('Running Baseline Bert') 
    
    bTformer = bert.Baseline(2e-5, 1, notesTrain, notesTest, labels, labels_test, 10)
    
    bTformer.makeInput()
    
    outMetrics = bTformer.run_baseline()
    '''
    
    '''
    print('Running MultiModal Model')
    
    MM = MultiModal(10, np.expand_dims(time_series,-1), np.expand_dims(time_series_test,-1), notesTrain, notesTest, [float(label) for label in labels], [float(label) for label in labels_test], model, 50)
    
    lossList = MM.train()
    
    ts.plot_value(lossList, "Number of Batches", "Loss", "Loss over Time")
    
    acc, AUPRC = MM.test()
    
    ts.plot_value(acc, "Number of Batches", "Accuracy", "Accuracy over Time")
    ts.plot_value(AUPRC, 'Number of Batches', 'AUPRC', 'AUPRC over Time')
    '''

if __name__ == '__main__':
    main()
    