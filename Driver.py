# loard required libraries
from scipy.sparse.construct import rand
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 
import pandas as pd 

c_v = pd.read_csv(r'C:\Users\Emily\Downloads\congressional_voting.csv', index_col=False)
c_v_array = c_v.to_numpy()

X = c_v_array[:,0:15] # contain the first 16 columns of the array (the feature being used to predict the category)
y = c_v_array[:,16] # consists the 17th column of the array, the categories 

# split the data into 70% training data and 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)

# train the perceptron
ppn.fit(X_train, y_train)

# apply the trained perceptron on the X data to make predicts for the y test data 
y_pred = ppn.predict(X_test)

print(y_pred) # predicted y test data
print(y_test) # true y test data 

# view the accuracy of the model, which is: 1- (observations predicted wrong / total observations)
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred)) 