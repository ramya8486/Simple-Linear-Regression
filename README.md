# Simple-Linear-Regression

Objective: The objective of the task was to predict students' scores based on their studying hours, using Supervised ML (Linear Regression with Python Scikit-Learn)

Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
Import the dataset
data = pd.read_csv("E:/GRIP Intern/Input.csv")
data
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
10	7.7	85
11	5.9	62
12	4.5	41
13	3.3	42
14	1.1	17
15	8.9	95
16	2.5	30
17	1.9	24
18	6.1	67
19	7.4	69
20	2.7	30
21	4.8	54
22	3.8	35
23	6.9	76
24	7.8	86
# To view top 5 data

data.head()
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
# Mark the response variable (y) and the predictor variable (x)

x=data.iloc[:,0].values
y=data.iloc[:,1].values
x,y
(array([2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5,
        3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8]),
 array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30,
        24, 67, 69, 30, 54, 35, 76, 86], dtype=int64))
To observe the distribution of data using scatter plot
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.scatter(x,y,color="red",marker="*")
<matplotlib.collections.PathCollection at 0x2d3377b69a0>
![image](https://github.com/ramya8486/Simple-Linear-Regression/assets/106485468/f4a870ad-2937-40bc-99aa-fa288f8df487)


Splitting the dataset
# Splitting data into training data and test data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
Fitting Simple Linear regression Model
regressor =LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train)
LinearRegression()
x_test
array([1.5, 3.2, 7.4, 2.5, 5.9])
y_test
array([20, 27, 69, 30, 62], dtype=int64)
# Compare the predicted value
regressor.predict(x_test[0].reshape(-1,1))
array([16.88414476])
Coeffcient and Intercept
# output

m = print(regressor.coef_)
b = print(regressor.intercept_)
[9.77580339]
2.48367340537321
y_pred =regressor.predict(x.reshape(-1,1))
y_pred
array([26.79480124, 52.56250809, 33.73226078, 86.25874013, 36.70545772,
       16.88414476, 93.19619966, 56.52677068, 84.27660883, 28.77693254,
       78.33021494, 60.49103328, 46.6161142 , 34.72332643, 12.91988217,
       90.22300272, 26.79480124, 20.84840735, 62.47316457, 75.357018  ,
       28.77693254, 49.58931115, 39.67865467, 70.40168976, 79.32128059])
R Squared value
# R squared value


r2_score(y,y_pred)
0.9526947647057274
To predict the percentage for 9.25 hours
# To predict the percentage for 9.25 hours

hours = np.array([9.25])
hours = hours.reshape(-1,1)
Pred_hour=regressor.predict(hours)
Pred_hour
array([93.69173249])
 
