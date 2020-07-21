import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_test)

#training simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)


#visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()



#Visualising the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()