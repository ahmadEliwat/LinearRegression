# LinearRegression
Linear Regression Project 
This is a Linear Regression Project and i upload two files (Code and Result)
Task 1: Data Preparation
Loading Data:

Import necessary libraries (numpy, pandas, matplotlib, seaborn).
Load the dataset using pandas and display the first few rows.
Data Cleaning:

Check the shape of the dataset.
Replace numeric origin codes with string labels (America, European, Japanies).
Drop the 'car name' column.
Identify and handle missing values in the 'horsepower' column.
Convert 'horsepower' to a numeric type after handling missing values.
Create dummy variables for the 'origin' column.
Exploratory Data Analysis:

Display summary statistics of the dataset.
Visualize relationships between variables using pair plots.
Task 2: Model Building and Training
Preparing Data for Modeling:

Separate the target variable (mpg) from the features.
Split the data into training and testing sets with a 30% training size and a random state of 23.
Model Training:

Initialize and train a LinearRegression model using the training data.
Display the coefficients and intercept of the trained model.
Interpret the coefficients and intercept in the context of the features.
Task 3: Model Evaluation
Making Predictions:

Use the trained model to make predictions on the test set.
Predict the mpg for a new data point [1, 55, 5000, 205, 2, 0, 5, 4, 1].
Model Performance:

Calculate and display the Mean Absolute Error (MAE) of the model on the test set.
Calculate and display the Root Mean Square Error (RMSE) of the model on the test set.
Additional Questions for Understanding
Data Handling:

Why is it important to replace missing values, and how does it affect the model?
What are the advantages of converting categorical variables into dummy/indicator variables?
Model Interpretation:

How would you interpret the coefficients of the linear regression model?
Discuss the impact of multicollinearity on the model, if any.
Model Evaluation:

What does the MAE tell you about the model’s performance?
How does the RMSE differ from MAE, and why is it important to consider both?
