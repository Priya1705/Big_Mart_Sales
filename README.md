# Big_Mart_Sales

BigMart sales dataset consists of 2013 sales data for 1559 products across 10 different outlets in different cities.This model helps BigMart understand the properties of products and stores that play an important role in increasing their overall sales.

1- data_prep.py  - prepares the dataset by converting data to numerical form and label encoding.
                   input = Train.csv
                   output = data_final.csv
2- missing_data.py  - removes issing values from the dataset using mean strategy.
                   input = data_final.csv
                   output = dataset.csv
3- naive.py   -   uses naive-bayes to model the data and do prediction.
4- svm.py   -     uses svm to model the data and do prediction. 
5- LinearRegression.py   -   uses Linear Regression to model the data and do prediction.
6- knn     -    uses knn to model the data and do prediction.
7- decision_tree.py   -   uses Decision Tree to model the data and do prediction. Uses gini index and entropy.
7- dt2.py   -   uses Decision Tree to model the data and do prediction. Uses gini index and entropy by default.
