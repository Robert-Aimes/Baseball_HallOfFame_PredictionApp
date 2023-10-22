#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[116]:


#load datasets into dataframes
pitchers = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\pitchers_official.csv')

batters = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\batting_official.csv')


# In[117]:


#plot pitcher ERA distribution

plt.hist(pitchers['Average of ERA'], bins=14, edgecolor="k", alpha=0.7, range=(0,10))
plt.title('Distribution of ERAs')
plt.xlabel('ERA')
plt.ylabel('Count')
plt.xlim(0, 10)  # Setting the x-axis limits
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


# In[118]:


# Convert the columns to datetime format with errors='coerce'
pitchers['finalGame'] = pd.to_datetime(pitchers['finalGame'], errors='coerce')
pitchers['debut'] = pd.to_datetime(pitchers['debut'], errors='coerce')

# Handle rows with NaT (optional, for instance, you can drop them)
pitchers.dropna(subset=['finalGame', 'debut'], inplace=True)

# Calculate the number of years between the two dates
pitchers['Years Played'] = (pitchers['finalGame'] - pitchers['debut']).dt.days / 365.25

# Round it if you need whole years
pitchers['Years Played'] = pitchers['Years Played'].round()


# In[119]:


pitchers.head()


# In[120]:


# Scatter plot of ERA/Years Played
plt.figure(figsize=(10, 6))
plt.scatter(pitchers['Years Played'], pitchers['Average of ERA'], alpha=0.5)
plt.title('ERA by Years Played')
plt.xlabel('Years Played')
plt.ylabel('ERA')
plt.grid(True)
plt.show()


# In[121]:


batters.info()


# In[122]:


plt.hist(batters['AVG'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of AVG')
plt.xlabel('AVG')
plt.ylabel('Number of Players')

# Explicitly set tick positions and labels
plt.xticks([0.100, 0.200, 0.300], ['0.100', '0.200', '0.300'])

plt.show()



# In[123]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:






# In[ ]:






# In[ ]:





# In[124]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Drop non-numeric columns or columns you feel are not necessary for the prediction:
drop_columns = ['Full Name', 'birthYear', 'birthMonth', 'birthDay', 'birthCountry', 
                'birthState', 'birthCity', 'nameFirst', 'nameLast', 'nameGiven', 
                'debut', 'finalGame', 'retroID', 'bbrefID']
pitchers = pitchers.drop(columns=drop_columns)

# List of columns for which you want to create new features
columns_to_average = [
    "Sum of GS", "Sum of CG", "Sum of SHO", "Sum of SV", "Sum of IPouts", 
    "Sum of ER", "Sum of H", "Sum of HR", "Sum of BB", "Sum of SO", 
    "Sum of BAOpp", "Sum of IBB", "Sum of WP", "Sum of HBP", "Sum of BK", 
    "Sum of BFP", "Sum of GF", "Sum of R", "Sum of SH", "Sum of SF", 
    "Sum of GIDP"
]

# For each column in the list, create a new feature based on average per 'Years Played'
for column in columns_to_average:
    # Remove 'Sum of ' from the column name and append '_per_year'
    new_column_name = f"{column.replace('Sum of ', '')}_per_year"
    pitchers[new_column_name] = pitchers[column] / pitchers["Years Played"]

# Drop the 'Years Played' column
pitchers.drop('Years Played', axis=1, inplace=True)

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



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[125]:


#Preidction model with current pitchers. Reads in csv and preps it for the model prediction

current_pitchers = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\pitching_current_players.csv')

# Convert the columns to datetime format with errors='coerce'
current_pitchers['finalGame'] = pd.to_datetime(current_pitchers['finalGame'], errors='coerce')
current_pitchers['debut'] = pd.to_datetime(current_pitchers['debut'], errors='coerce')

# Handle rows with NaT (optional, for instance, you can drop them)
current_pitchers.dropna(subset=['finalGame', 'debut'], inplace=True)


# Calculate the number of years between the two dates
current_pitchers['Years Played'] = (current_pitchers['finalGame'] - current_pitchers['debut']).dt.days / 365.25

# Round it if you need whole years
current_pitchers['Years Played'] = current_pitchers['Years Played'].round()

# List of columns for which you want to create new features
columns_to_average = [
    "Sum of GS", "Sum of CG", "Sum of SHO", "Sum of SV", "Sum of IPouts", 
    "Sum of ER", "Sum of H", "Sum of HR", "Sum of BB", "Sum of SO", 
    "Sum of BAOpp", "Sum of IBB", "Sum of WP", "Sum of HBP", "Sum of BK", 
    "Sum of BFP", "Sum of GF", "Sum of R", "Sum of SH", "Sum of SF", 
    "Sum of GIDP"
]

# For each column in the list, create a new feature based on average per 'Years Played'
for column in columns_to_average:
    # Remove 'Sum of ' from the column name and append '_per_year'
    new_column_name = f"{column.replace('Sum of ', '')}_per_year"
    current_pitchers[new_column_name] = current_pitchers[column] / current_pitchers["Years Played"]

# Check the first few rows of the DataFrame to confirm changes
print(current_pitchers.head())


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



# In[126]:


filtered_rows = current_pitchers[current_pitchers['predicted_induction'] == 'Y']
print(filtered_rows[['playerID', 'predicted_induction']])


# In[127]:


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


# In[128]:


filtered_results = current_pitchers[current_pitchers['induction_probability'] > 0.1]
print(filtered_results)

import matplotlib.pyplot as plt

plt.hist(current_pitchers['induction_probability'], bins=20, edgecolor='k', alpha=0.7)
plt.title('Distribution of Induction Probabilities')
plt.xlabel('Probability')
plt.ylabel('Number of Players')
plt.show()



# In[129]:


#Batting model creation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data
# data = pd.read_csv('path_to_file.csv') # Uncomment this if loading from CSV
bdata = batters

# Replace '#DIV/0!' with NaN (Not a Number)
bdata.replace('#DIV/0!', float('nan'), inplace=True)

# Now, decide how to handle these NaN values. There are a few common strategies:
# 1. Fill NaN with zero:
bdata.fillna(0, inplace=True)

for column in ['bats', 'throws']:
    bdata[column] = bdata[column].astype(str)
# Convert 'bats' and 'throws' using label encoding
label_encoders = {}
for column in ['bats', 'throws']:
    le = LabelEncoder()
    bdata[column] = le.fit_transform(bdata[column])
    label_encoders[column] = le
    
# Calculate new features based on 'num_years' for bdata
bdata['hits_per_year'] = bdata['H'] / bdata['num_years']
bdata['runs_per_year'] = bdata['R'] / bdata['num_years']
bdata['rbis_per_year'] = bdata['RBI'] / bdata['num_years']
bdata['hr_per_year'] = bdata['HR'] / bdata['num_years']

# If you want to drop 'num_years' from bdata:
bdata.drop('num_years', axis=1, inplace=True)


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

# Predict
# predictions = clf.predict(new_data) # Use this for new data


# In[130]:


# Load the current player data
# Assuming the data is in a CSV file:
current_batting_player_data = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\batting_current_players.csv')

# Replace '#DIV/0!' with NaN (Not a Number)
current_batting_player_data.replace('#DIV/0!', float('nan'), inplace=True)

# Fill NaN with zero:
current_batting_player_data.fillna(0, inplace=True)

# Convert 'bats' and 'throws' using label encoding
for column in ['bats', 'throws']:
    le = label_encoders[column]
    
    # Handle unknown labels
    def encode_with_fallback(s):
        try:
            return le.transform([s])[0]
        except ValueError:
            # If the value isn't known, transform it to a default value.
            # Here, I'm using the first class of the encoder as a default.
            return le.transform([le.classes_[0]])[0]
    
    current_batting_player_data[column] = current_batting_player_data[column].apply(encode_with_fallback)

# Calculate new features based on 'num_years'
current_batting_player_data['hits_per_year'] = current_batting_player_data['H'] / current_batting_player_data['num_years']
current_batting_player_data['runs_per_year'] = current_batting_player_data['R'] / current_batting_player_data['num_years']
current_batting_player_data['rbis_per_year'] = current_batting_player_data['RBI'] / current_batting_player_data['num_years']
current_batting_player_data['hr_per_year'] = current_batting_player_data['HR'] / current_batting_player_data['num_years']


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


# In[131]:


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


# In[132]:


filtered_batting_results = current_batting_player_data[current_batting_player_data['induction_probability'] > 0.1]
print(filtered_batting_results)


# In[133]:


importances = b_clf.feature_importances_
features = b_features.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
print(importance_df.sort_values(by='Importance', ascending=False))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




