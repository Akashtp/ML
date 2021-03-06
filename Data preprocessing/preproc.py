import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Reading data from csv file and splitting the values to independant and dependant sets
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Taking care of missing data by adding mean value
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#Encoding independant variables
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])] ,remainder = 'passthrough' )
X = np.array(ct.fit_transform(X))


#Encoding dependant variable
le = LabelEncoder()
y = le.fit_transform(y)


#Splitting dataset into trainset and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2, random_state = 1)  


#Feature scaling
sc = StandardScaler()
X_train[ :, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[ :, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)
print(y_train)
print(y_test)