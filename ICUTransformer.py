#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:23:48 2020

@author: sollyboukman
"""

import numpy as np
import tensorflow as tf

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, num_heads=10):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        if embedding_dimension % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embedding_dimension} should be divisible by number of heads = {num_heads}")
        
        self.proj_dim = embedding_dimension // num_heads
        self.q_dense = tf.keras.layers.Dense(embedding_dimension)
        self.k_dense = tf.keras.layers.Dense(embedding_dimension)
        self.v_dense = tf.keras.layers.Dense(embedding_dimension)
        self.combine_heads = tf.keras.layers.Dense(embedding_dimension)
        
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        key_dim = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(key_dim)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0,2,1,3])
    
    def call(self, inputs):
        # input shape = [batch_size, seq_length, embedding_dimension]
        batch_size = tf.shape(inputs)[0]
        query = self.q_dense(inputs) # (batch_size, seq_length, proj_dim)
        key = self.k_dense(inputs) # (batch_size, seq_length, proj_dim)
        value = self.v_dense(inputs) # (batch_size, seq_length, proj_dim)
        
        query = self.separate_heads(query, batch_size) # (batch_size, num_heads, seq_length, proj_dim)
        key = self.separate_heads(key, batch_size) # (batch_size, num_heads, seq_length, proj_dim)
        value = self.separate_heads(value, batch_size) # (batch_size, num_heads, seq_length, proj_dim)
        
        
        attention, weights = self.attention(query, key, value)
        
        attention = tf.transpose(attention, perm=[0,2,1,3]) # (batch_size, seq_length, num_heads, proj_dim)
        
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embedding_dimension)) # (batch_size, seq_length, embedding_dimension)
        
        output = self.combine_heads(concat_attention) # (batch_size, seq_length, embedding_dimension)
        
        return output
    
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, num_heads, hidden_size, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.atten = MultiHeadSelfAttention(embedding_dimension, num_heads)
        self.hidden_size = hidden_size
        self.feed_forwards = tf.keras.Sequential([tf.keras.layers.Dense(self.hidden_size), tf.keras.layers.Dense(embedding_dimension)])
        
        self.norm_layer1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout_layer1 = tf.keras.layers.Dropout(rate)
        self.dropout_layer2 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, inputs, training):
        attention_out = self.atten(inputs)
        attention_out = self.dropout_layer1(attention_out, training=training)
        out1 = self.norm_layer1(inputs + attention_out)
        ff_out = self.feed_forwards(out1)
        ff_out = self.dropout_layer2(ff_out, training=training)
        return self.norm_layer2(out1 + ff_out)
    
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embedding_dimension):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dimension)
        self.pos_embeddings = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embedding_dimension)
                                    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_embeddings(positions)
        x = self.token_embeddings(x)
        return x + positions
    
class Notes_Transformer(tf.keras.Model):
    def __init__(self, maxlen, vocab_size, num_heads):
        super(Notes_Transformer, self).__init__()
        
        self.embedding_dimension = 100
        self.hidden_size = 50
        
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, self.embedding_dimension)
        
        self.transformer_block = TransformerBlock(self.embedding_dimension, num_heads, self.hidden_size)
        
        self.glb_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Droupout(0.1)
        self.dense1 = tf.keras.layers.Dense(20, activation="relu")
        self.dense2 = tf.keras.layers.Dense(2, activation="softmax")

        
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
    def call(self, inputs):
        
        inputs = tf.keras.layers.Input(shape=(self.maxlen,))
        
        x = self.embedding_layer(inputs)
        
        x = self.transformer_block(x)
        
        x = self.glb_avg_pool()(x)
        
        x = self.dropout1(x)
        
        x = self.dense1(x)
        
        x = self.dropout2(x)
        
        outputs = self.dense2(x) # will not do this when concatenating with message data
        
        return outputs
        
'''

This runs through a full binary classification instance with transformer, this will be used
as a baseline for our model in itself. Our actual implementation will not output the a vector
to be concatenated to LSTM timeseries output

"""
      


         
        
        
        
