from flask import Flask, request, jsonify
import pandas as pd
from model import generate_forecast  # The function from Step 1
import json

app = Flask(__name__)

# Load the dataset once
df = pd.read_csv('final/ph_dengue_cases2016-2020.csv') 

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    region = data.get('region')
    months = int(data.get('months', 12))

    if region is None or months <= 0:
        return jsonify({'error': 'Invalid region or months'}), 400

    try:
        result = generate_forecast(df, region, months)
        # Format the output nicely
        result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
        result_json = result.to_dict(orient='records')
        return jsonify(result_json)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

