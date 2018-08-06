# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 18:19:20 2018

@author: Aditya Vartak
"""

# preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
labelencoder_X2=LabelEncoder()
z=X
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])

X[:, 2] = labelencoder_X2.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling is very important in deep learning
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#part 2 ANN

import keras
from keras.models import Sequential #to initialise a NN as sequence of layers
from keras.layers import Dense #to create layers of NN

#initialise NN

classifier=Sequential()

#add input layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))#choosing no of nodes in hodden layer is depend on user,However Parameter tuning can be used for optimal no of nodes.Also begineers can put it as avg of no of inputs and outputs

#hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))#can also use softmax which is sigmoid applied to dependent variable having more than 2 categories

#compiling the ANN

classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])#loss function is logarithmic and if the outcome is binary then use binary_crossentropy else for categorical data use categorical_crossentropy

#fitting Ann to training set

classifier.fit(X_train,y_train,batch_size=10,epochs=100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred > 0.5)


#check if given customer would leave the bank or not
new_pred=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred=(new_pred > 0.5)#which comes as false hence customer wont leave the bank
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#using k fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_ANN_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))#choosing no of nodes in hodden layer is depend on user,However Parameter tuning can be used for optimal no of nodes.Also begineers can put it as avg of no of inputs and outputs
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))#can also use softmax which is sigmoid applied to dependent variable having more than 2 categories
    classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])#loss function is logarithmic and if the outcome is binary then use binary_crossentropy else for categorical data use categorical_crossentropy
    return classifier

classifier=KerasClassifier(build_fn=build_ANN_classifier , batch_size=10,epochs=100 )
accuracies=cross_val_score(classifier,X_train,y_train,cv = 10,n_jobs = -1)# n_jobs means no of parallel jobs ,-1 means ALL CPU utilisation

    
# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_