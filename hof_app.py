from flask import Flask, render_template, jsonify, request
import pandas as pd

app = Flask(__name__, static_folder='static', static_url_path='/static')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_players', methods=['GET'])
def get_players():
    # Read the CSV file
    data = pd.read_csv('./data/hof_predictions.csv')

    # Extract player names and IDs
    players = [{"id": row["playerID"], "name": row["Full Name"]} for _, row in data.iterrows()]

    # Return the data as JSON
    return jsonify(players)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the POST request
    player_type = request.form['player_type']
    player_id = request.form['player_id']

    # Here, you would integrate your machine learning model to get the prediction.
    # For the sake of this demonstration, I'm returning a placeholder response.
    # Replace this with your actual model logic.
    result = "High Chance"

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)

