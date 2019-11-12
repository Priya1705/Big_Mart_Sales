from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset=pd.read_csv('dataset.csv')

x=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

lin_reg=LinearRegression()
lin_reg.fit(x_train, y_train)
predicted=lin_reg.predict(x_test)

test_rmse=(np.sqrt(mean_squared_error(y_test, predicted)))
test_r2=r2_score(y_test, predicted)

print(test_rmse)
print(test_r2)