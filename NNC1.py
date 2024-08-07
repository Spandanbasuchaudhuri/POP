import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the CSV file
file_path = 'F:\Data\Data for satellite Project.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
columns_to_drop = ['Name', 'Column1']
data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

# Function to remove outliers using the IQR method
def remove_outliers(df):
    cleaned_df = df.copy()
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)]
    return cleaned_df

# Remove outliers
data_no_outliers = remove_outliers(data_cleaned)

# Normalize the data
scaler = MinMaxScaler()
numeric_columns = data_no_outliers.select_dtypes(include=[np.number]).columns
numeric_columns = numeric_columns.drop('RG')  # Exclude the target variable
normalized_data = data_no_outliers.copy()
normalized_data[numeric_columns] = scaler.fit_transform(data_no_outliers[numeric_columns])

# Data Preparation
features = normalized_data.drop(columns=['RG'])
target = normalized_data['RG']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Predict and evaluate the model
y_pred_nn = model.predict(X_test).flatten()

# Define the adjusted accuracy function
def adjusted_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    y_range = y_true.max() - y_true.min()
    accuracy_value = (1 - mae / y_range) * 100
    return accuracy_value

adjusted_accuracy_nn = adjusted_accuracy(y_test, y_pred_nn)

print(f"Adjusted Accuracy of Neural Network: {adjusted_accuracy_nn:.2f}%")

# Function to take user input for predicting RG
def predict_rg(model, scaler, feature_columns):
    user_input = {}
    for column in feature_columns:
        value = float(input(f"Enter value for {column}: "))
        user_input[column] = value
    
    user_input_df = pd.DataFrame(user_input, index=[0])
    normalized_input = scaler.transform(user_input_df)
    prediction = model.predict(normalized_input).flatten()[0]
    return prediction

# Predict RG based on user input
feature_columns = features.columns
predicted_rg = predict_rg(model, scaler, feature_columns)
print(f"Predicted RG value: {predicted_rg:.2f}")
