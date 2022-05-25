import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data_classification.csv', header=None)

features = data.values[: ,:2]   
labels = data.values[: ,2]      


array_1 = np.ones((100,1))
features_con = np.concatenate((array_1, features), axis=1)
"""
true_x = []
true_y = []
false_x = []
false_y = []
for item in data.values:
    if item[2] == 1.:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])

plt.scatter(true_x, true_y, marker='o', c='b')
plt.scatter(false_x, false_y, marker='s', c='r')
plt.show()
"""
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sep(p):
    if p >= 0.5:
        return 1
    else:
        return 0

def predict(features_con, weights):
    z = np.dot(features_con, weights)  
    return sigmoid(z) 

def cost_function(features_con, labels, weights):
    n = len(labels)
    predictions = predict(features_con, weights)    
                                                
                                               
    cost_class1 = -labels*np.log(predictions)  
    cost_class2 = -(1-labels)*np.log(1 - predictions)   
    cost = cost_class1 + cost_class2    
    return cost.sum()/n 

def update_weight(features_con, labels, weights, learning_rate):
    n = len(labels) 

    predictions = predict(features_con, weights)
    gd = np.dot(features_con.T, (predictions - labels)) 
    gd = gd/n
    gd = gd*learning_rate
    weights = weights - gd
    return weights

def train(features_con, labels, weights, learning_rate, iter):
    cost_his = []
    for i in range(iter):
        weights = update_weight(features_con, labels, weights, learning_rate)
        cost = cost_function(features_con, labels, weights)
        cost_his.append(cost)

    return weights, cost_his

x1 = float(input('enter feature 1: '))
x2 = float(input('enter feature 2: '))


weights, cost = train(features_con, labels, [0.5, 0.2, 0.05], 0.2, 5000)

print('predict value:')


p = sep(predict([1, x1, x2], weights))

print(p)
