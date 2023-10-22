import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#read in datasets
#former pitchers
pitchers = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\former_pitchers_prepped.csv')
#current pitchers
current_pitchers = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\current_pitchers_prepped.csv')
#former batters
bdata = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\former_batters_prepped.csv')
#current batters
current_batting_player_data = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\current_batters_prepped.csv')


#Pither model creation
X = pitchers.drop('inducted', axis=1)
y = pitchers['inducted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Replace infinite values with NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# One-Hot Encoding
X_train = pd.get_dummies(X_train, columns=['bats', 'throws'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['bats', 'throws'], drop_first=True)

# It's essential to ensure that both X_train and X_test have the same columns after one-hot encoding.
# So, we align them.
X_train, X_test = X_train.align(X_test, axis=1, fill_value=0)

# Drop the 'playerID' column before imputation
X_train = X_train.drop(columns=['playerID'])
X_test = X_test.drop(columns=['playerID'])

# Store column names before imputation
column_names = X_train.columns

# Impute missing and infinite values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Convert X_train and X_test back to DataFrames using the stored column names
X_train = pd.DataFrame(X_train, columns=column_names)
X_test = pd.DataFrame(X_test, columns=column_names)

# Apply the scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Model for pitchers
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make pitcher Prediction
y_pred = clf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))


#Current Pitcher model prediction

# Drop non-numeric columns or columns you feel are not necessary for the prediction:
drop_columns = ['Full Name', 'birthYear', 'birthMonth', 'birthDay', 'birthCountry', 
                'birthState', 'birthCity', 'nameFirst', 'nameLast', 'nameGiven', 
                'debut', 'finalGame', 'retroID', 'bbrefID']
# 1. Drop unnecessary columns, including 'inducted'
current_pitchers = current_pitchers.drop(columns=drop_columns + ['inducted'])

# 2. Drop 'playerID' for preprocessing (we'll retain it later for identification)
X_current = current_pitchers.drop('playerID', axis=1)

# 3. One-Hot Encoding
X_current = pd.get_dummies(X_current, columns=['bats', 'throws'], drop_first=True)

X_train = pd.DataFrame(X_train, columns=column_names)
X_train_df = X_train.copy()

# Align the dataset with the original training data columns (using X_train_df)
X_current, _ = X_current.align(X_train_df, axis=1, fill_value=0)

X_current = X_current[X_train_df.columns]

print(X_current.columns)
print(X_train_df.columns)

# 1. Identify and replace infinite values
X_current.replace([np.inf, -np.inf], np.nan, inplace=True)

# 5. Impute missing values
X_current_imputed = imputer.transform(X_current)  # Using the previously defined 'imputer'

# Convert back to DataFrame using the stored column names
X_current_imputed = pd.DataFrame(X_current_imputed, columns=column_names)

# 6. Scale the data
X_current_scaled = scaler.transform(X_current_imputed) # Using the previously defined 'scaler'

# 7. Make predictions using the trained Random Forest model
current_predictions = clf.predict(X_current_scaled)

# 8. Add predictions back to the 'current_players' DataFrame
current_pitchers['predicted_induction'] = current_predictions

# 9. Display predictions for each player
print(current_pitchers[['playerID', 'predicted_induction']])

infinite_columns = X_current.columns.to_series()[np.isinf(X_current).any()]
print(infinite_columns)

filtered_rows = current_pitchers[current_pitchers['predicted_induction'] == 'Y']
print(filtered_rows[['playerID', 'predicted_induction']])

# Using the trained Random Forest model
current_probabilities = clf.predict_proba(X_current_scaled)

# Let's consider the probability of the positive class (assuming it's the second column)
induction_probabilities = current_probabilities[:, 1]

# You can then decide a threshold, let's say 0.5 (50%)
current_predictions = (induction_probabilities >= 0.5).astype(int)

# Add probabilities and predictions back to the 'current_pitchers' DataFrame
current_pitchers['induction_probability'] = induction_probabilities
current_pitchers['predicted_induction'] = current_predictions

# Display predictions and probabilities for each player
print(current_pitchers[['playerID', 'induction_probability', 'predicted_induction']])


filtered_results = current_pitchers[current_pitchers['induction_probability'] > 0.1]
print(filtered_results)


#Batting model creation
# Split the data
b_features = bdata.drop(columns=['playerID', 'Full Name', 'inducted', 'retroID', 'bbrefID', 'debut', 'finalGame', 'stint'])
# Check for non-numeric data in each column
for col in b_features.columns:
    try:
        b_features[col].astype(float)
    except ValueError:
        print(f"Column '{col}' has non-numeric data.")
b_target = bdata['inducted']
b_X_train, b_X_test, b_y_train, b_y_test = train_test_split(b_features, b_target, test_size=0.2, random_state=42)



# Train a Decision Tree model
b_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
b_clf.fit(b_X_train, b_y_train)

# Evaluate the model
b_y_pred = b_clf.predict(b_X_test)
b_accuracy = accuracy_score(b_y_test, b_y_pred)
print(f"Model Accuracy: {b_accuracy}")




#Current Batter model prediction
# Drop the columns not used in training
current_b_features = current_batting_player_data.drop(columns=['playerID', 'Full Name', 'inducted', 'retroID', 'bbrefID', 'debut', 'finalGame', 'num_years', 'stint'])

# If there's any missing columns in current_features that were present in the training data, align them
# This fills any missing columns in the current data with 0 (since that's what you used for NaN values)
current_b_features, _ = current_b_features.align(b_X_train, axis=1, fill_value=0)

# Make predictions
batting_predictions = b_clf.predict(current_b_features)

# If you want to add the predictions to your current player data DataFrame:
current_batting_player_data['predicted_induction'] = batting_predictions

print(current_batting_player_data[['playerID', 'Full Name', 'predicted_induction']])



# Using the trained Decision Tree model
batting_probabilities = b_clf.predict_proba(current_b_features)

# Let's consider the probability of the positive class (assuming it's the second column)
induction_probabilities_batting = batting_probabilities[:, 1]

# You can then decide a threshold, let's say 0.5 (50%)
batting_predictions_probabilistic = (induction_probabilities_batting >= 0.5).astype(int)

# Add probabilities and predictions back to the 'current_batting_player_data' DataFrame
current_batting_player_data['induction_probability'] = induction_probabilities_batting
current_batting_player_data['predicted_induction_probabilistic'] = batting_predictions_probabilistic

# Display predictions and probabilities for each player
print(current_batting_player_data[['playerID', 'Full Name', 'induction_probability', 'predicted_induction_probabilistic']])

filtered_batting_results = current_batting_player_data[current_batting_player_data['induction_probability'] > 0.1]
print(filtered_batting_results)


importances = b_clf.feature_importances_
features = b_features.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
print(importance_df.sort_values(by='Importance', ascending=False))
