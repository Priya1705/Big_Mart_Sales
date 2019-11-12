from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

dataset=pd.read_csv('data_final.csv')

# 0 for columns and 1 for rows
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset.iloc[:,:])
dataset.iloc[:,:] = imputer.transform(dataset.iloc[:,:]) 

dataset.to_csv("dataset.csv",index=False, header=True)