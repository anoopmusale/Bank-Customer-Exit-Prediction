# -*- coding: utf-8 -*-
#Data Preprocessing 
#Importing Libraries

#Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Making ANN

#Importing Keras Libraries


'''WITHOUT K_FOLD
#Initialising ANN
classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))

#Second Hidden layer
classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu'))

#Third Hidden layer
classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu'))

#Output Layer
classifier.add(Dense(units = 1,kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling ANN
classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

#Fitting training set to ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.8)

#Predict New data
new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred>0.8)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
#Evaluating ANN
#K-Fold
#Dropout for preventing/removing overfitting

'''
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 1,kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()
'''

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 8,kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 1,kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer=optimizer, loss= 'binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25,32],
              'epochs' : [100,500],
              'optimizer' : ['adam','rmsprop']}
gridSearch = GridSearchCV(estimator = classifier, param_grid= parameters, scoring = 'accuracy', cv =10)
gridSearch=gridSearch.fit(X_train,y_train)
best_para = gridSearch.best_params_
best_acc = gridSearch.best_score_






'''print("Accuracy Percent: ", ((cm[0][0]+cm[1][1])/(X_test.size/11))*100)'''