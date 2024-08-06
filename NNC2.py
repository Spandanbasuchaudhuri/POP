import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = 'F:\Data\Data for satellite Project.csv'
df = pd.read_csv(file_path)

# Drop the specified columns
df = df.drop(columns=['Column1', 'CHIRPS', 'Name'])

# Calculate the IQR for each numerical column to identify outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Remove outliers from the dataframe using the IQR method
df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Separate features and target before normalization
X = df_no_outliers.drop(columns=['RG'])
y = df_no_outliers['RG']

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
df_normalized = pd.DataFrame(X_normalized, columns=X.columns)
df_normalized['RG'] = y.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_normalized.drop(columns=['RG']), df_normalized['RG'], test_size=0.2, random_state=42)

# Build the Neural Network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Make predictions on the test set
y_pred_nn = model.predict(X_test)

# Evaluate the Neural Network model's performance
mae_nn = mean_absolute_error(y_test, y_pred_nn)
range_of_actuals_nn = y_test.max() - y_test.min()
accuracy_mae_nn = (1 - mae_nn / range_of_actuals_nn) * 100

print("Mean Absolute Error (MAE):", mae_nn)
print("Accuracy using MAE:", accuracy_mae_nn, "%")

# Get user input for prediction
user_input = {}
for column in X.columns:
    value = float(input(f"Enter value for {column}: "))
    user_input[column] = value

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Ensure the order of columns in user input matches the training data
user_input_df = user_input_df[X.columns]

# Normalize the user input
user_input_normalized = scaler.transform(user_input_df)

# Make prediction
prediction = model.predict(user_input_normalized).flatten()
print(f"Predicted RG value: {prediction[0]:.2f}")
