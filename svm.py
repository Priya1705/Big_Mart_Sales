from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

dataset=pd.read_csv('dataset.csv')

x=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

support = svm.LinearSVC(random_state=20)
support.fit(x_train, y_train)
predicted= support.predict(x_test)

score=accuracy_score(y_test,predicted)
print("Your Model Accuracy is", score)