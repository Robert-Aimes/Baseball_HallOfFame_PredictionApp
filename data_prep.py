
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


#load datasets into dataframes
pitchers = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\pitchers_official.csv')

batters = pd.read_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\batting_official.csv')

#Prepping former pitcher data for model and feature engineering
# Convert the columns to datetime format with errors='coerce'
pitchers['finalGame'] = pd.to_datetime(pitchers['finalGame'], errors='coerce')
pitchers['debut'] = pd.to_datetime(pitchers['debut'], errors='coerce')

# Handle rows with NaT (optional, for instance, you can drop them)
pitchers.dropna(subset=['finalGame', 'debut'], inplace=True)

# Calculate the number of years between the two dates
pitchers['Years Played'] = (pitchers['finalGame'] - pitchers['debut']).dt.days / 365.25

# Round it if you need whole years
pitchers['Years Played'] = pitchers['Years Played'].round()

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

#write out prepped data
pitchers.to_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\former_pitchers_prepped.csv', index=False)


#Prepping current pitchers for prediction
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

#write out prepped data
current_pitchers.to_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\current_pitchers_prepped.csv', index=False)



#Prep former Batter data for model training
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

#write out prepped data
bdata.to_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\former_batters_prepped.csv', index=False)

#Prep current batters for prediction
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

#write out prepped file
current_batting_player_data.to_csv(r'C:\Users\robai\OneDrive\Documents\MLB datasets\current_batters_prepped.csv', index=False)
