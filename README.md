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
    - Use the `shape` attribute to determine the number of rows and columns in the dataset.
    ```python
    cData.shape
    ```

2. **Replace numeric origin codes with string labels (America, European, Japanies).**
    - Use the `replace` method to convert numeric codes in the 'origin' column to their respective string labels.
    ```python
    cData.replace({1: "America", 2: "European", 3: "Japanies"}, inplace=True)
    ```

3. **Drop the 'car name' column.**
    - Use the `drop` method to remove the 'car name' column as it is not needed for the analysis.
    ```python
    cData = cData.drop('car name', axis=1)
    ```

4. **Identify and handle missing values in the 'horsepower' column.**
    - First, check for missing values using `isnull` and `sum` methods.
    - Then, replace any non-numeric values ('?') in the 'horsepower' column with NaN and fill these with the median of the column.
    ```python
    cData.isnull().sum()
    hpIsDigit = pd.DataFrame(cData.horsepower.str.isdigit())
    cData = cData.replace('?', np.nan)
    cData["horsepower"] = cData["horsepower"].fillna(cData["horsepower"].median()).astype('float64')
    ```

5. **Create dummy variables for the 'origin' column.**
    - Convert the 'origin' column to dummy/indicator variables using `get_dummies`.
    ```python
    cData = pd.get_dummies(cData, columns=['origin'])
    ```

#### Exploratory Data Analysis:

1. **Display summary statistics of the dataset.**
    - Use the `describe` method to generate summary statistics for the dataset.
    ```python
    cData.describe()
    ```

2. **Visualize relationships between variables using pair plots.**
    - Use `seaborn`'s `pairplot` to visualize the relationships between different numerical variables in the dataset.
    ```python
    cData_attr = cData.iloc[:, 0:7]
    sns.pairplot(cData_attr, diag_kind='kde')
    ```

### Task 2: Model Building and Training

#### Preparing Data for Modeling:

1. **Separate the target variable (`mpg`) from the features.**
    - Assign the `mpg` column to `y` and all other columns to `X`.
    ```python
    y = cData['mpg']
    X = cData.drop('mpg', axis=1)
    ```

2. **Split the data into training and testing sets with a 30% training size and a random state of 23.**
    - Use `train_test_split` from `sklearn` to split the data.
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.30, random_state=23)
    ```

#### Model Training:

1. **Initialize and train a `LinearRegression` model using the training data.**
    - Create an instance of `LinearRegression` and fit it to the training data.
    ```python
    from sklearn.linear_model import LinearRegression
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    ```

2. **Display the coefficients and intercept of the trained model.**
    - Print out the coefficients and intercept of the model.
    ```python
    for idx, column in enumerate(X_train.columns):
        print(f"The coefficient of {column} is {regression_model.coef_[idx]}")
    
    print(f"The intercept for our model is {regression_model.intercept_}")
    ```

3. **Interpret the coefficients and intercept in the context of the features.**
    - Discuss what each coefficient means in relation to the feature it represents.
    ```markdown
    The coefficients represent the change in the target variable (mpg) for a one-unit change in the feature, holding all other features constant. The intercept is the expected value of mpg when all features are zero.
    ```

### Task 3: Model Evaluation

#### Making Predictions:

1. **Use the trained model to make predictions on the test set.**
    - Use the `predict` method to generate predictions for the test set.
    ```python
    predictions = regression_model.predict(X_test)
    ```

2. **Predict the `mpg` for a new data point `[1, 55, 5000, 205, 2, 0, 5, 4, 1]`.**
    - Reshape the new data point to match the input shape of the model and use `predict`.
    ```python
    new_data_point = np.array([1, 55, 5000, 205, 2, 0, 5, 4, 1]).reshape(1, -1)
    predicted_mpg = regression_model.predict(new_data_point)
    ```

#### Model Performance:

1. **Calculate and display the Mean Absolute Error (MAE) of the model on the test set.**
    - Use `mean_absolute_error` from `sklearn.metrics` to calculate MAE.
    ```python
    from sklearn import metrics
    mae = metrics.mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
    ```

2. **Calculate and display the Root Mean Square Error (RMSE) of the model on the test set.**
    - Use `mean_squared_error` from `sklearn.metrics` to calculate RMSE.
    ```python
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    print(f"Root Mean Square Error: {rmse}")
    ```

### Additional Questions for Understanding

#### Data Handling:

1. **Why is it important to replace missing values, and how does it affect the model?**
    ```markdown
    Replacing missing values is crucial because missing data can lead to incorrect analysis and poor model performance. Models cannot handle missing values directly, so imputing them ensures that all data points are utilized effectively, leading to more accurate predictions.
    ```

2. **What are the advantages of converting categorical variables into dummy/indicator variables?**
    ```markdown
    Converting categorical variables into dummy variables allows the model to interpret and process them correctly. This transformation enables categorical data to be used in mathematical computations, ensuring that each category is treated as a separate feature.
    ```

#### Model Interpretation:

1. **How would you interpret the coefficients of the linear regression model?**
    ```markdown
    Each coefficient represents the change in the target variable (mpg) for a one-unit change in the respective feature, assuming all other features are held constant. A positive coefficient indicates a direct relationship, while a negative coefficient indicates an inverse relationship.
    ```

2. **Discuss the impact of multicollinearity on the model, if any.**
    ```markdown
    Multicollinearity occurs when independent variables are highly correlated, leading to unreliable estimates of coefficients. It can inflate the standard errors and make it difficult to determine the individual effect of each feature. Detecting and addressing multicollinearity is essential for accurate model interpretation.
    ```

#### Model Evaluation:

1. **What does the MAE tell you about the model’s performance?**
    ```markdown
    MAE measures the average absolute errors between predicted and actual values. It provides a straightforward interpretation of prediction accuracy, indicating the average magnitude of errors in the model's predictions.
    ```

2. **How does the RMSE differ from MAE, and why is it important to consider both?**
    ```markdown
    RMSE measures the square root of the average squared differences between predicted and actual values. Unlike MAE, RMSE gives more weight to larger errors. Considering both MAE and RMSE provides a comprehensive view of model performance, balancing the average error magnitude and sensitivity to larger errors.
    ```
