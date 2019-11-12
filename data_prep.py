from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer

dataset=pd.read_csv('Train.csv')

dataset=dataset.iloc[0:,1:]

dataset=pd.DataFrame(dataset);
dataset.to_csv("data.csv",index=False, header=False)

x=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1]

fat_C={
	"Low Fat": 0,
	"low fat": 0,
	"LF":0,
	"Regular":1,
	"reg":1
}

size_d={
	"Small":0,
	"Medium":1,
	"High":2
}

# ---------------------------------change values of fat columns using the above dictionary made cz of spelling difference-----------------------------
for i in range(8523):
	if(x.iloc[i,1] in fat_C):
		x.iloc[i,1]=fat_C[x.iloc[i,1]]



labelencoder_3 = LabelEncoder()
labelencoder_5 = LabelEncoder()
labelencoder_8 = LabelEncoder()
labelencoder_9 = LabelEncoder()
x.iloc[:, 3] = labelencoder_3.fit_transform(x.iloc[:, 3])
x.iloc[:, 5] = labelencoder_5.fit_transform(x.iloc[:, 5])
x.iloc[:, 8] = labelencoder_8.fit_transform(x.iloc[:, 8])
x.iloc[:, 9] = labelencoder_9.fit_transform(x.iloc[:, 9])


for i in range(8523):
	if(x.iloc[i,7] in size_d):
		x.iloc[i,7]=size_d[x.iloc[i,7]]

x.to_csv("data_final.csv",index=False, header=True)
