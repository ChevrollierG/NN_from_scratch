# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:44:43 2024

@author: guill
"""

import random
from pprint import pprint
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.cifar10 as cifar10

class Layer():
    def __init__(self):
        pass
    
    def forward(self, in_data):
        pass
    
    def backward(self, gradient, lr):
        pass
    
class Conv2D(Layer):
    def __init__(self, data_depth, kernel_depth, kernel_size):
        self.weights = []
        for i in range(kernel_depth):
            self.weights.append(np.array([np.array([np.array([round(random.random(), 2) for j in range(kernel_size)]) for k in range(kernel_size)]) for l in range(data_depth)]))
        self.weights = np.array(self.weights)
                
    def forward(self, in_data):
        result = np.zeros((np.shape(self.weights)[0], np.shape(in_data)[1], np.shape(in_data)[2]))
        padding = 1
        self.in_data = in_data
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                result[i] += np.array(convolution(in_data[j], self.weights[i][j], padding))
        print(result)
        return result
    
    def backward(self, gradient, lr):
        weights_gradient = np.zeros(np.shape(self.weights))
        input_gradient = np.zeros(np.shape(self.in_data))
        
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                weights_gradient[i][j] = convolution(self.in_data[j], gradient[i], 1)
                input_gradient[j] += np.array(convolution(gradient[i], rotate180(self.weights[i][j]), 1))
                
        self.weights -= lr * weights_gradient
        return input_gradient
    
class MaxPooling(Layer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
    
    def forward(self, in_data):
        result = []
        if(len(in_data[0])%self.kernel_size != 0):
            raise Exception("The size of the image should be divisible by the kernel size")
        self.pos_max = []
        for i in range(len(in_data)):
            result.append([])
            self.pos_max.append([])
            for j in range(0, len(in_data[i]), self.kernel_size):
                if((len(in_data[i]) - j) >= self.kernel_size):
                    result[i].append([])
                    self.pos_max[i].append([])
                    for k in range(0, len(in_data[i][j]), self.kernel_size):
                        if((len(in_data[i][j]) - k) >= self.kernel_size):
                            result[i][int(j/self.kernel_size)].append(in_data[i][j][k])
                            self.pos_max[i][int(j/self.kernel_size)].append((j,k))
                            for l in range(self.kernel_size):
                                for m in range(self.kernel_size):
                                    if(in_data[i][j + l][k + m] > result[i][int(j/self.kernel_size)][int(k/self.kernel_size)]):
                                        result[i][int(j/self.kernel_size)][int(k/self.kernel_size)] = in_data[i][j + l][k + m]
                                        self.pos_max[i][int(j/self.kernel_size)][int(k/self.kernel_size)] = (j+m,k+m)
        return result
    
    def backward(self, gradient, lr):
        result = []
        for i in range(len(self.pos_max)):
            result.append([])
            for j in range(self.kernel_size**2):
                result[i].append([])
                for k in range(self.kernel_size**2):
                    if(self.pos_max[i][int(j/self.kernel_size)][int(k/self.kernel_size)][0] == j and self.pos_max[i][int(j/self.kernel_size)][int(k/self.kernel_size)][1] == k):
                        result[i][j].append(gradient[i][int(j/self.kernel_size)][int(k/self.kernel_size)])
                    else:
                        result[i][j].append(0)
        return result
            
    
class ReLu(Layer):
    def __init__(self):
        pass
    
    def forward(self, in_data):
        result = []
        self.in_data = in_data
        for i in range(len(in_data)):
            result.append([])
            for j in range(len(in_data[i])):
                result[i].append([])
                for k in range(len(in_data[i][j])):
                    result[i][j].append(max(0, in_data[i][j][k]))
        return result
    
    def backward(self, gradient, lr):
        result = []
        for i in range(len(self.in_data)):
            result.append([])
            for j in range(len(self.in_data[i])):
                result[i].append([])
                for k in range(len(self.in_data[i][j])):
                    if(self.in_data[i][j][k] >= 0):
                        result[i][j].append(gradient[i][j][k])
                    else:
                        result[i][j].append(0)
        return result
    
class SoftMax(Layer):
    def __init__(self):
        pass
    
    def forward(self, in_data):
        memory = 0
        for i in range(len(in_data)):
            memory += math.exp(in_data[i])
            
        result = []
        for i in range(len(in_data)):
            result.append(math.exp(in_data[i]) / memory)
        self.output = result
        return result
    
    def backward(self, gradient, lr):
        result = []
        for i in range(len(self.output)):
            result.append([])
            for j in range(len(self.output)):
                if(i == j):
                    result[i].append(self.output[i] * (1-self.output[i]))
                else:
                    result[i].append(-self.output[i] * self.output[j])
        result = np.dot(result, gradient)
        return result
                
    
class Flatten(Layer):
    def __init__(self):
        pass
    
    def forward(self, in_data):
        result = []
        self.input_size = len(in_data[0])
        for i in range(len(in_data)):
            for j in range(len(in_data[i])):
                for k in range(len(in_data[i][j])):
                    result.append(in_data[i][j][k])
        return result
    
    def backward(self, gradient, lr):
        result = [[]]
        for i in gradient:
            if(len(result[len(result)-1]) == 0):
                result[len(result)-1].append([i])
            elif(len(result[len(result)-1][len(result[len(result)-1])-1]) == self.input_size):
                if(len(result[len(result)-1]) == self.input_size):
                    result.append([[i]])
                else:
                    result[len(result)-1].append([i])
            else:
                result[len(result)-1][len(result[len(result)-1])-1].append(i)
        return result
    
class Dense(Layer):
    def __init__(self, nb_neurons, input_size):
        self.weights = []
        self.bias = []
        for i in range(nb_neurons):
            self.weights.append(np.array([round(random.random(), 2) for j in range(input_size)]))
            self.bias.append(round(random.random(), 2))
    
    def forward(self, in_data):
        result = []
        self.in_data = in_data
        for i in range(len(self.weights)):
            result.append(self.bias[i])
            for j in range(len(self.weights[i])):
                result[i] += self.weights[i][j] * in_data[j]
        return result
    
    def backward(self, gradient, lr):
        weights_gradient = np.dot(gradient, self.in_data)
        input_gradient = np.dot(gradient, self.weights)
        
        self.weights -= lr * weights_gradient
        self.bias -= lr * np.array(gradient)
        
        return input_gradient
    
class BatchNormalization(Layer):
    def __init__(self):
        pass
    
    def forward(self, in_data):
        result = []
        self.in_data = in_data
        for i in range(len(self.weights)):
            result.append(self.bias[i])
            for j in range(len(self.weights[i])):
                result[i] += self.weights[i][j] * in_data[j]
        return result
    
    def backward(self, gradient, lr):
        weights_gradient = np.dot(gradient, self.in_data)
        input_gradient = np.dot(gradient, self.weights)
        
        self.weights -= lr * weights_gradient
        self.bias -= lr * np.array(gradient)
        
        return input_gradient
                    
def CrossEntropyLoss(y_pred, y_truth):
    loss = 0
    for i in range(len(y_truth)):
        loss += y_truth[i] * math.log(y_pred[i]) + (1-y_truth[i]) * math.log(1-y_pred[i])
    loss = -loss/len(y_truth)
    return loss

def CrossEntropyLoss_prime(y_pred, y_truth):
    result = []
    for i in range(len(y_pred)):
        result.append(((1-y_truth[i])/(1-y_pred[i]) - y_truth[i]/y_pred[i])/len(y_pred))
    return result

def rotate180(data):
    result = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[i])):
            result[i][j] = data[len(data) - 1 - i][len(data[i]) - 1 - j]
    return result

def convolution(data, kernel, padding):
    result = np.zeros(np.shape(data))
    for i in range(0, len(data)+2*padding):
        if((len(data) + 2*padding - i) >= len(kernel)):
            for j in range(0, len(data[i])+2*padding):
                if((len(data[i]) + 2*padding - j) >= len(kernel)):
                    for k in range(len(kernel)):
                        if(i-padding+k >= 0 and i-padding+k <= len(data)-1):
                            for l in range(len(kernel[k])):
                                if(j-padding+l >= 0 and j-padding+l <= len(data[i])-1):
                                    result[i][j] += kernel[k][l] * data[i - padding + k][j - padding + l]
    return result

def reshape_images(data):
    result = np.zeros((np.shape(data)[0], np.shape(data)[3], np.shape(data)[1], np.shape(data)[2]))
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                for l in range(len(data[i][j][k])):
                    result[i][l][j][k] += round(data[i][j][k][l] / 255.0, 2)
    return result
            

def dataset(count):
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()
    Y = []
    for i in range(count):
        Y.append(np.array([1 if Ytrain[i]==j else 0 for j in range(10)]))
    
    return (Xtrain[:count], np.array(Y)), (Xtest, Ytest)
    
    
def model(input_depth, y_size):
    result = [Conv2D(input_depth, 5, 3),
              ReLu(),
              MaxPooling(2),
              Conv2D(5, 5, 3),
              ReLu(),
              MaxPooling(2),
              Conv2D(5, 5, 3),
              ReLu(),
              MaxPooling(2),
              Flatten(),
              Dense(10, 80),
              SoftMax()]
    return result

def train_model(data_X, data_Y, model):
    EPOCH = 10
    lr = 0.01
    
    for i in range(EPOCH):
        error = 0
        for j in range(len(data_X)):
            input_data = data_X[i]
            for k in range(len(model)):
                input_data = model[k].forward(input_data)
                #print(input_data[0])
                
            error += CrossEntropyLoss(input_data, data_Y[j])
            gradient = CrossEntropyLoss_prime(input_data, data_Y[j])
            
            for k in range(len(model)):
                gradient = model[k].backward(gradient, lr)
                
        print("Epoch: ", i, ", Loss: ", error)
        
    return model



if __name__ == '__main__':
    dataset_size = 10
    nb_class = 10
    image_type = 3
    (Xtrain, Ytrain), (Xtest, Ytest) = dataset(dataset_size)
    print("Dataset loaded")
    Xtrain = reshape_images(Xtrain)
    model = model(image_type, nb_class)
    model = train_model(Xtrain, Ytrain, model)
    