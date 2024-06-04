## Tasks

### Task 1: Data Preparation

#### Loading Data:

1. **Import necessary libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`).**
    ```python
    import numpy as np   
    import pandas as pd    
    import matplotlib.pyplot as plt 
    %matplotlib inline 
    import seaborn as sns
    ```

2. **Load the dataset using `pandas` and display the first few rows.**
    ```python
    cData = pd.read_csv('C:\\Users\\TECH\\Desktop\\Data Science\\Machine Learning\\auto-mpg.csv')
    cData.head()
    ```

#### Data Cleaning:

1. **Check the shape of the dataset.**
    ```python
    cData.shape
    ```

2. **Replace numeric origin codes with string labels (America, European, Japanies).**
    ```python
    cData.replace({1: "America", 2: "European", 3: "Japanies"}, inplace=True)
    ```

3. **Drop the 'car name' column.**
    ```python
    cData = cData.drop('car name', axis=1)
    ```

4. **Identify and handle missing values in the 'horsepower' column.**
    ```python
    cData.isnull().sum()
    hpIsDigit = pd.DataFrame(cData.horsepower.str.isdigit())
    cData = cData.replace('?', np.nan)
    cData["horsepower"] = cData["horsepower"].fillna(cData["horsepower"].median()).astype('float64')
    ```

5. **Create dummy variables for the 'origin' column.**
    ```python
    cData = pd.get_dummies(cData, columns=['origin'])
    ```

#### Exploratory Data Analysis:

1. **Display summary statistics of the dataset.**
    ```python
    cData.describe()
    ```

2. **Visualize relationships between variables using pair plots.**
    ```python
    cData_attr = cData.iloc[:, 0:7]
    sns.pairplot(cData_attr, diag_kind='kde')
    ```

### Task 2: Model Building and Training

#### Preparing Data for Modeling:

1. **Separate the target variable (`mpg`) from the features.**
    ```python
    y = cData['mpg']
    X = cData.drop('mpg', axis=1)
    ```

2. **Split the data into training and testing sets with a 30% training size and a random state of 23.**
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.30, random_state=23)
    ```

#### Model Training:

1. **Initialize and train a `LinearRegression` model using the training data.**
    ```python
    from sklearn.linear_model import LinearRegression
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    ```

2. **Display the coefficients and intercept of the trained model.**
    ```python
    for idx, column in enumerate(X_train.columns):
        print(f"The coefficient of {column} is {regression_model.coef_[idx]}")
    
    print(f"The intercept for our model is {regression_model.intercept_}")
    ```

### Task 3: Model Evaluation

#### Making Predictions:

1. **Use the trained model to make predictions on the test set.**
    ```python
    predictions = regression_model.predict(X_test)
    ```

2. **Predict the `mpg` for a new data point `[1, 55, 5000, 205, 2, 0, 5, 4, 1]`.**
    ```python
    new_data_point = np.array([1, 55, 5000, 205, 2, 0, 5, 4, 1]).reshape(1, -1)
    predicted_mpg = regression_model.predict(new_data_point)
    ```

#### Model Performance:

1. **Calculate and display the Mean Absolute Error (MAE) of the model on the test set.**
    ```python
    from sklearn import metrics
    mae = metrics.mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
    ```

2. **Calculate and display the Root Mean Square Error (RMSE) of the model on the test set.**
    ```python
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    print(f"Root Mean Square Error: {rmse}")
    ```

### Additional Questions for Understanding

#### Data Handling:

1. **Why is it important to replace missing values, and how does it affect the model?**
2. **What are the advantages of converting categorical variables into dummy/indicator variables?**

#### Model Interpretation:

1. **How would you interpret the coefficients of the linear regression model?**
2. **Discuss the impact of multicollinearity on the model, if any.**

#### Model Evaluation:

1. **What does the MAE tell you about the model’s performance?**
2. **How does the RMSE differ from MAE, and why is it important to consider both?**
