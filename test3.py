import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

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

# Step 7: Define hyperparameter grids for each model

# 1. Support Vector Regression (SVR)
svr_params = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2]
}

svr_model = GridSearchCV(SVR(), svr_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
svr_model.fit(X_train, y_train)
print(f"Best SVR Hyperparameters: {svr_model.best_params_}")

# 2. Decision Tree Regressor
dt_params = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

dt_model = GridSearchCV(DecisionTreeRegressor(), dt_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
dt_model.fit(X_train, y_train)
print(f"Best Decision Tree Hyperparameters: {dt_model.best_params_}")

# 3. Random Forest Regressor
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
rf_model.fit(X_train, y_train)
print(f"Best Random Forest Hyperparameters: {rf_model.best_params_}")

# 4. Gradient Boosting Regressor
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10]
}

gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
gb_model.fit(X_train, y_train)
print(f"Best Gradient Boosting Hyperparameters: {gb_model.best_params_}")

# 5. XGBoost Regressor
xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10]
}

xgb_model = GridSearchCV(xgb.XGBRegressor(random_state=42, objective='reg:squarederror'), xgb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
xgb_model.fit(X_train, y_train)
print(f"Best XGBoost Hyperparameters: {xgb_model.best_params_}")

# 6. LightGBM Regressor
lgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 127],
    'boosting_type': ['gbdt', 'dart']
}

lgb_model = GridSearchCV(lgb.LGBMRegressor(random_state=42), lgb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
lgb_model.fit(X_train, y_train)
print(f"Best LightGBM Hyperparameters: {lgb_model.best_params_}")

# Step 8: Evaluate all models on the test set
models = {
    'SVR': svr_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    max_error = y_test.max() - y_test.min()
    accuracy = (1 - (mae / max_error)) * 100
    print(f"{name} - Mean Absolute Error: {mae}, Accuracy: {accuracy}%")
