import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Define the adjusted accuracy function
def adjusted_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    y_range = y_true.max() - y_true.min()
    accuracy_value = (1 - mae / y_range) * 100
    return accuracy_value

# Load the CSV file
file_path = 'path_to_your_csv_file.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
columns_to_drop = ['name', 'column1']
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
normalized_data = data_no_outliers.copy()
normalized_data[numeric_columns] = scaler.fit_transform(data_no_outliers[numeric_columns])

# Data Preparation
features = normalized_data.drop(columns=['RG', 'Column1', 'Name'])
target = normalized_data['RG']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
adjusted_accuracy_value = adjusted_accuracy(y_test, y_pred)

print(f"Adjusted Accuracy of Gradient Boosting Regressor: {adjusted_accuracy_value:.2f}%")
