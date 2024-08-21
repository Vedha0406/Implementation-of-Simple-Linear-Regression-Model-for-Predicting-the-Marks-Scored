# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)  
*/
```

## Output:
Dataset

output

![dataset](https://github.com/user-attachments/assets/4e02772d-eed6-43f6-8932-0fb0751720bd)

Head Values

output

![head](https://github.com/user-attachments/assets/191c1a02-3486-426e-bffe-b3d05d51d91c)

Tail Values

output

![tail](https://github.com/user-attachments/assets/883da906-50b9-4da8-b79c-5c90291141d5)

X and Y values

output

![xyvalues](https://github.com/user-attachments/assets/cfec8252-ad31-4206-9782-2527fb05b918)

Predication values of X and Y
output
![predict ](https://github.com/user-attachments/assets/de6de4b6-6868-4369-983f-59adeb5d424e)


MSE,MAE and RMSE

output

![values](https://github.com/user-attachments/assets/66ae6c5b-b714-4316-aafe-92225c2e8a3f)

Training Set

output

![train](https://github.com/user-attachments/assets/813aac8b-3431-4385-8afb-96ddfffd3926)

Testing Set

output

![test](https://github.com/user-attachments/assets/9a5018c1-cb1b-468e-8c06-d607b60aa780)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
