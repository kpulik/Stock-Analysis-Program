from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd

# Load the data
data = pd.read_csv('stock_data.csv')

# Choose the predictor variables, here we'll use all available variables except for the target variable
predictors = data.drop('Stock Price', axis=1)

# Choose the target variable
target = data['Stock Price']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=0)

# Create a linear regression model
regressor = LinearRegression()  

# Train the model using the training data
regressor.fit(X_train, y_train)

# Use the model to make predictions using the test data
y_pred = regressor.predict(X_test)
