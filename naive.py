import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 


dataset=pd.read_csv("dataset.csv")
x=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model= GaussianNB()
model.fit(x_train, y_train)
predicted=model.predict(x_test)

score=accuracy_score(y_test,predicted)
print("Your Model Accuracy is", score)