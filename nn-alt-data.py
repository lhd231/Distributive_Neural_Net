# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:48:56 2016

@author: LHD
"""

import numpy as np
import random
import pandas as pd
from sklearn import datasets



class neuralNetwork:
    
    model = {}
    
    i = 0
    
    Z = np.array
    X = np.array
    y = np.array
    num_examples = 1
    nn_input_dim = 2 #Dimensionality of the input layer
    nn_output_dim = 2 #Dimensionality of output layer
    
    epsilon = .01 #learning rate
    reg_lambda = .01 #regularization strength
    
    W1 = np.array #Weights between the input layer and hidden layer
    b1 = np.array
    W2 = np.array #Weights between hidden and output
    b2 = np.array
    
    a1 = np.array
    def __init__(self,D, y):
        self.Z = D
        self.X = self.Z[0]
        self.y = y
        
    def calculate_loss(self,model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        #forward propagation
        z1 = self.X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #calculate the loss
        correct_logprobs = -np.log(probs[range(self.num_examples), y])
        data_loss = np.sum(correct_logprobs)
        #Add regularization term to loss
        data_loss += self.reg_lambda/2 * (np.sum(np.square(W1))) + np.sum(np.square(W2))
        return 1./self.num_examples * data_loss
        
    def predict(self,model, x):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
        
    #Create the model by setting the number of nodes per layer and
    #and randomize the weights with a given seed
    def create_model(self,seed,nn_hdim):
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, nn_hdim))
        self.W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        self.b2 = np.zeros((1, self.nn_output_dim))
    
    #Do a single test and return the delta 3
    def forward_propagation(self, s):
        #Given a number i, pick that member of the dataset
        #should be random
        #np.random.seed(s)

        self.i = s#random.randint(0,len(self.Z)-1)
      
        self.X = np.array(self.Z[self.i])[np.newaxis]
                
        # Forward propagation
        z1 = self.X.dot(self.W1) + self.b1
        self.a1 = np.tanh(z1)
        z2 = self.a1.dot(self.W2) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        delta3 = probs
        delta3[range(self.num_examples), self.y[self.i]] -= 1        
        return delta3
    
    def back_propagation(self, delta3):
        # Backpropagation


        self.X = np.array(self.Z[self.i])[np.newaxis]
        dW2 = (self.a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(self.W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(self.X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += self.reg_lambda * self.W2
        dW1 += self.reg_lambda * self.W1
        
        # Gradient descent parameter update
        self.W1 += -self.epsilon * dW1
        self.b1 += -self.epsilon * db1
        self.W2 += -self.epsilon * dW2
        self.b2 += -self.epsilon * db2
        
        # Assign new parameters to the model
        self.model = { 'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
            
            
    def get_model(self):
        return self.model

       
np.random.seed(0)

	# read data as array
img_data = np.asarray(pd.read_csv("train.csv", sep=',', header=None, low_memory=False))
	# get labels
labels = np.asarray(img_data[1:,0], np.dtype(int))
	# omit header and labels
img_data = img_data[1:,1:]
imgs = []	
y = []
print len(img_data)
print "Building data set with only 8s and 1s"
for i in range(len(img_data)):
    if labels[i] == 8:
         y.append(1)
         img = img_data[i]
         imgs.append(img)
    elif labels[i] == 1:
        y.append(-1)
        img = img_data[i]
        imgs.append(img)

for i in range(len(y)):
    print y[i]
X = np.asarray(imgs, np.dtype(float))
print len(y)
print len(X)
print "after load"
X, y = datasets.make_moons(200, noise=0.20)
Z1 = np.asarray(X[0:len(X)/2])
Z2 = np.asarray(X[len(X)/2:len(X)])
y1 = np.asarray(y[0:len(y)/2])
y2 = np.asarray(y[len(y)/2:len(y)])
print "after parsing"

answer = 0.0
NN = neuralNetwork(Z1,y1)
NN.create_model(0,3)
print "after net 1"
NN2 = neuralNetwork(Z2,y2)
NN2.create_model(0,3)
print "after net 2"
model = {}
print len(X)
for i in range(len(Z1)):
    d1 = NN.forward_propagation(i)
    d2 = NN2.forward_propagation(i)
    
    avg = (d1+d2)/2
    NN.back_propagation(avg)
    NN2.back_propagation(avg)
print "after process"
model = NN2.get_model()

for i in range(len(Z2)):
    if NN2.predict(model,Z2[i]) == y2[i]:
        answer += 1
print answer / len(Z2)

