import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn import tree


dataset=pd.read_csv("dataset.csv")
x=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model= tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
predicted=model.predict(x_test)

score=accuracy_score(y_test,predicted)
print("Your Model Accuracy is", score)