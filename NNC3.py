import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load the data
file_path = 'F:\Data\Data for satellite Project.csv'
data = pd.read_csv(file_path)

# Step 2: Drop unnecessary columns
data_cleaned = data.drop(columns=['Name', 'Column1', 'IMERG'])

# Step 3: Remove outliers using the IQR method
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

data_no_outliers = remove_outliers_iqr(data_cleaned)

# Step 4: Separate features and target
X = data_no_outliers.drop(columns=['RG'])
y = data_no_outliers['RG']

# Step 5: Normalize the features (without the target)
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 7: Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Step 8: Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Step 9: Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Step 10: Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Calculate accuracy as a percentage
max_error = y_test.max() - y_test.min()
accuracy = (1 - (mae / max_error)) * 100

print(f"Neural Network - Mean Absolute Error: {mae}")
print(f"Neural Network - Accuracy: {accuracy}%")

# Step 11: Function to make predictions using dynamic user inputs
def get_user_input():
    user_input = {}
    for feature in X.columns:
        value = float(input(f"Enter value for {feature}: "))
        user_input[feature] = value
    return user_input

def predict_RG():
    input_data = get_user_input()
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_normalized = scaler.transform(input_df)
    prediction = model.predict(input_normalized)
    return prediction[0]

# Example usage
predicted_RG = predict_RG()
print(f"Predicted RG value using Neural Network: {predicted_RG}")
