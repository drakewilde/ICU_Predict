
from transformers import TFBertForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import timedelta
import re

class PreprocessNote():
    '''
    Class for preprocessing note data 
    -loads note data csv
    -finds the latest note for the first 48 hours of a stay for a given HADM_ID
    '''
    
    def __init__(self, filename):
        self.file = filename
        self.stay_fstNoteTime = {}
        self.stayLastNote = {}
        self.timeframe = timedelta(hours = 48)
        
        
    def readNotes(self):
        '''
        Reads in the clinical note file instantiated in the model
        Converts the the chartime to type datetime

        '''
        df = pd.read_csv(self.file, sep=",") #, nrows=30
        
        df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"], format='%Y-%m-%d %H')
        
        return df
    
    def populateStayTime(self, df):
        '''
        Finds the first time a note was taken for a HADM_ID
        '''
        
        uniqueStays = set(df["HADM_ID"].to_list())

        for stay in uniqueStays:
           
           stayData = df[df["HADM_ID"] == stay]
           
           firstNoteTime = stayData[stayData["CHARTTIME"] == min(stayData["CHARTTIME"])]["CHARTTIME"].iloc[0]
           
           self.stay_fstNoteTime[stay] = firstNoteTime
           
           
    def avgNoteLen(self):
        '''
        Returns average length of a note
        '''
        totStays = len(self.stayLastNote)
        totNoteLen = 0
        for key in self.stayLastNote:
            totNoteLen += len(self.stayLastNote[key].split())
            
        return totNoteLen / totStays
            
            
            
    def pupulateLastNote(self, df):
        '''
        Gets the last note for a HADM_ID
        '''
        
        for key in self.stay_fstNoteTime:
            
            data = df[(df["CHARTTIME"] >= self.stay_fstNoteTime[key]) & (df["CHARTTIME"] <= (self.stay_fstNoteTime[key] + self.timeframe))]
            
            latest_note = data[data["CHARTTIME"] == max(data["CHARTTIME"])]["TEXT"].iloc[0]
            
            self.stayLastNote[key] = latest_note
            
            
            
          
    def cleanNotes(self):
        '''
        Cleans note for each HADM_ID in the last note dictionary

        '''
        
        for key in self.stayLastNote:
            
            self.stayLastNote[key] = re.sub('\\[(.*?)\\]', '', self.stayLastNote[key])
            self.stayLastNote[key] = re.sub('[0-9]+\.', '', self.stayLastNote[key])
            self.stayLastNote[key] = re.sub('dr\.', 'doctor', self.stayLastNote[key])
            self.stayLastNote[key] = re.sub('m\.d\.', 'md', self.stayLastNote[key])
            self.stayLastNote[key] = re.sub('admission date:', '', self.stayLastNote[key])
            self.stayLastNote[key] = re.sub('discharge date:', '', self.stayLastNote[key])
            self.stayLastNote[key] = re.sub('--|__|==', '', self.stayLastNote[key])
            
            self.stayLastNote[key] = self.stayLastNote[key].replace('\n', ' ')
            self.stayLastNote[key] = self.stayLastNote[key].replace('\r', ' ')
            self.stayLastNote[key] = self.stayLastNote[key].strip()
            self.stayLastNote[key] = self.stayLastNote[key].lower()
            self.stayLastNote[key] = self.stayLastNote[key].lower() #+ ' [SEP]'
            
            #next part is to test forward pass
            self.stayLastNote[key] = ' '.join(self.stayLastNote[key].split()[0:50])
            
            
            
            

class Baseline():
    '''
    Class for Bio_ClinicalBERT transformer
    -functionality for training and testing an individual note data model
    -functionality to output hidden layers and embeddings for multimodal model
    '''
    def __init__(self, learning_rate, num_epochs, train_encoded, test_encoded, trainLabels, testLabels, batchSz):
        '''
        Parameters
        ----------
        learning_rate : float
            given learning rate for pre-trained BERT model
        num_epochs : int
            number of epochs
        train_encoded : list if notes
            train note data can be encoded using make input or forward pass
        test_encoded : list of notes
            test note data can be encoded using make input or forward pass
        trainLabels : list of integers
            train data labels
        testLabels : list of integers
            test data labels
        batchSz : int
            given batch size (recomended 10, 16, or 32 maximum)

        Returns
        -------
        None.

        '''
        self.model = TFBertForSequenceClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', from_pt=True)
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.learningRate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=self.learningRate)
        self.epochs = num_epochs
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=[self.metric])
        self.train_data = train_encoded
        self.test_data = test_encoded
        self.trainLabels = trainLabels
        self.testLabels = testLabels
        self.batchSz = batchSz
        self.modelTraininput = None
        self.modelTestinput = None
        
    def run_baseline(self):
        '''
        Runs the sole clinical BERT transformer shows loss and accuracy updating in real time

        '''
        
        bert_history = self.model.fit(self.modelTraininput, batch_size=self.batchSz, epochs=self.epochs, validation_data=self.modelTestinput)
        
        return bert_history
    
    
    def forwardPass(self, inputs):
        '''
        Runs a forward pass on a batch of note data

        '''
        input_ids = []
        token_types = []
        attention_mask = []
       
        
        for note in inputs:#self.train_data: (changed to be generalized)
            
            input = self.convertToFeature(note)
            
            input_ids.append(input['input_ids'])
            token_types.append(input['token_type_ids'])
            attention_mask.append(input['attention_mask'])
           
            
        
        id_tensor = tf.convert_to_tensor(input_ids)
        token_type_tensor = tf.convert_to_tensor(token_types)
        attention_mask_tensor = tf.convert_to_tensor(attention_mask)
        print("calling forward pass")
        output = self.model.call(inputs=id_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_type_tensor, output_hidden_states=True)
        return output
    
    def convertToFeature(self, note):
        '''
        Converts a note to a feature using the instantiated tokenizer
        '''
        
        
        return self.tokenizer.encode_plus(note, add_special_tokens = True, max_length = 512, pad_to_max_length = True, return_attention_mask = True)
    
    def mapToDict(self, input_ids, attention_masks, token_type_ids, label):
        '''
        Returns a dictionary of the inputs provided
        '''
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_masks}, label
    
    def makeInput(self):
        '''
        Creates a tensorflow dataset for train and test data and saves them
        NOTE: MUST BE RUN BEFORE run_baseline()
        '''
        TrainInput_ids = []
        TrainToken_types = []
        TrainAttention_mask = []
        TrainLabel_list = []
        
        TestInput_ids = []
        TestToken_types = []
        TestAttention_mask = []
        TestLabel_list = []
        
        for i in range(0, len(self.trainLabels)):
            
            input = self.convertToFeature(self.train_data[i])
            
            TrainInput_ids.append(input['input_ids'])
            TrainToken_types.append(input['token_type_ids'])
            TrainAttention_mask.append(input['attention_mask'])
            TrainLabel_list.append([int(self.trainLabels[i])])
            
        print(TrainLabel_list)
            
        self.modelTraininput = tf.data.Dataset.from_tensor_slices((TrainInput_ids, TrainAttention_mask, TrainToken_types, TrainLabel_list)).map(self.mapToDict).batch(self.batchSz)
        #print(self.model_input)
        #print(label_list)
        
        for i in range(0, len(self.testLabels)):
            
            input = self.convertToFeature(self.test_data[i])
            
            TestInput_ids.append(input['input_ids'])
            TestToken_types.append(input['token_type_ids'])
            TestAttention_mask.append(input['attention_mask'])
            TestLabel_list.append([int(self.testLabels[i])])
            
                                   
        self.modelTestinput = tf.data.Dataset.from_tensor_slices((TestInput_ids, TestAttention_mask, TestToken_types, TestLabel_list)).map(self.mapToDict).batch(self.batchSz)
        
