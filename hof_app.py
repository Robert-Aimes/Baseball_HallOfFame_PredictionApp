from flask import Flask, render_template, jsonify, request
import pandas as pd

import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load the CSV file into a global DataFrame
data = pd.read_csv('./data/hof_predictions.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_players', methods=['GET'])
def get_players():
    # Extract player names, IDs, and types from the global DataFrame
    players = [{"id": row["playerID"], "name": row["Full Name"], "type": row["Position"]} for _, row in data.iterrows()]

    # Return the data as JSON
    return jsonify(players)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the POST request
    player_id = request.form['player_id']

    # Fetch the corresponding "HOF Percent Chance" for the given player ID
    hof_chance = data[data["playerID"] == player_id]["HOF Percent Chance"].values[0]

    return jsonify({"result": hof_chance})

if __name__ == '__main__':
    app.run(debug=True)

