# midterm-project-group-2-web-based
### Group Participants
***Group 2B: Web-based Machine Learning with Python***, 

consisting of:
  + [KANOKPATRA KANATAN](https://github.com/POSTTTT) (LEADER)
  + [MAHMUDJONOV ZOHIDJON MURODJON UGLI](https://github.com/zohidjon-m)
  + [GANBOLD OTGONBAATAR](Otgonbaatar)
  + [JASON CLARENCE](https://github.com/realjyce)

Contents below are the same as: **[Basic-design-projects-update.pdf](https://github.com/realjyce/midterm-project-group-2-web-based/blob/main/Basic%20design%20projects-Update.pdf)** and **[README](./README.md)**

## Project 2: Web-based Machine Learning App Guideline

The objective of this project is to design a web-based program (you can use this course for help: 
https://www.coursera.org/projects/machine-learning-streamlit-python) that, in addition to being 
user-friendly, includes the following tasks:
| **No.** | **Step-by-step Guidethrough** |
| --- | --- |
| 1. | Selecting input data (via CSV file) (Data.CSV) |
| 2. | Selecting independent and dependent data Independent data (all columns except the Flood column) Dependent data (Flood column) |
| 3. | Selecting the percentage ratio of training and testing data. The program should allow the selection of the following ratios for training and testing data: |
  | | o   90:10 (90% training data, 10% testing data) |
  | | o   80:20 (80% training data, 20% testing data) |
  | | o   70:30 (70% training data, 30% testing data) |
  | | o   60:40 (60% training data, 40% testing data) |
| 4. | Choosing a machine learning model for prediction on training and testing data: Machine learning models in regression mode (`Random Forest` or `XGBoost`) |
| 5. | Running the model on training and testing data and evaluating the results with the following metrics:|
| | `RMSE` |
| | `MAE` |
| | `R2` |
| 6. | Displaying a histogram of errors between actual (Flood) and predicted data in step 5 |
| 7. | Displaying the determination of the importance of criteria using the machine learning model |
| 8. |  Calling new data and predicting on this data: New data in CSV format is called initially. 
| | This data has two columns x and y (longitude and latitude) and other columns similar to independent data in step 2. After predicting the model on this data (introducing independent columns) using the prediction output and values of x and y, display a density heat map here (you can use MapBox). |

| Extra Credits |
| --- |
|o Steps 6 and 7 are considered extra credit. |
|o Choose a greater number of algorithms in step four.|
|o Display the area boundary in step 8 using https://mapshaper.org/ or Leaflet.|

## An Example of Different Steps for Executing a Machine Learning Algorithm in Python:
1. Import the Required Libraries:
```import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from permetrics.regression import RegressionMetric
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics
```
2. Load the Data:
```
ReadData = pd.read_csv('/content/drive/MyDrive/…../Data.csv')
```
4. Split the Data into Training and Test Sets:
```
X = ReadData.drop([Flood], axis = 1) 
y = ReadData [Flood]
4. Set the Data Ratio:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, 
random_state=1)
```
5. Create and Train the Model:
```
model = RandomForestRegressor(random_state = 1).fit(X_train, y_train)
```
7. Predict Using the Model on Training and Test Data:
```yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)
```
7. Evaluate the Results for Training and Test Data:
```
r2 = metrics.r2_score(y_train, yhat_train)
mae = metrics.mean_absolute_error(y_train, yhat_train)
mse = metrics.mean_squared_error(y_train, yhat_train)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, yhat_test)
mae = metrics.mean_absolute_error(y_test, yhat_test)
mse = metrics.mean_squared_error(y_test, yhat_test)
rmse = np.sqrt(mse)
```
8. Load New Data:
```
NewData = pd.read_csv('/content/drive/MyDrive/…../NewData.csv')
```
10. Predict Using the Model on New Data:
```
output_prediction = model.predict(Newdata)
```
12. Calculate the Error Histogram:
```error = y_test - y_test_pred
plt.hist(error, bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("Error Histogram")
plt.show()
```
