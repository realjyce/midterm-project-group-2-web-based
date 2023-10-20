# midterm-project-group-2-web-based
Group 2 Web-based Machine Learning with Python
Group 2 Section 2(Web-based) with the members of: Kanatan(Leader), Jason, Ollie, Zohidjon.
// Contents below are the same as the | Basic-design-projects-update.pdf |
//
Project 2: Web-based Machine Learning App
The objective of this project is to design a web-based program (you can use this course for help: 
https://www.coursera.org/projects/machine-learning-streamlit-python) that, in addition to being 
user-friendly, includes the following tasks:
1. Selecting input data (via CSV file) (Data.CSV)
2. Selecting independent and dependent data Independent data (all columns except the Flood 
column) Dependent data (Flood column)
3. Selecting the percentage ratio of training and testing data. The program should allow the 
selection of the following ratios for training and testing data:
o 90:10 (90% training data, 10% testing data)
o 80:20 (80% training data, 20% testing data)
o 70:30 (70% training data, 30% testing data)
o 60:40 (60% training data, 40% testing data)
4. Choosing a machine learning model for prediction on training and testing data: Machine 
learning models in regression mode (Random Forest or XGBoost)
5. Running the model on training and testing data and evaluating the results with the 
following metrics: RMSE, MAE, and R2
6. Displaying a histogram of errors between actual (Flood) and predicted data in step 5
7. Displaying the determination of the importance of criteria using the machine learning 
model
8. Calling new data and predicting on this data: New data in CSV format is called initially. 
This data has two columns x and y (longitude and latitude) and other columns similar to 
independent data in step 2. After predicting the model on this data (introducing independent 
columns) using the prediction output and values of x and y, display a density heat map here 
(you can use MapBox).
Extra Credit:
 Steps 6 and 7 are considered extra credit.
 Choose a greater number of algorithms in step four.
 Display the area boundary in step 8 using https://mapshaper.org/ or Leaflet.
