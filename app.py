from flask import Flask, jsonify, request
import pandas as pd
from model import generate_forecast  # your function from model.py

app = Flask(__name__)

# Load raw data once globally
df = pd.read_csv('dengue_data.csv')  # use your full dataset here

@app.route('/')
def home():
    return "<h2>Dengue Forecast API</h2><p>Use endpoint <code>/forecast?region=NCR&months=12</code></p>"

@app.route('/forecast', methods=['GET'])
def forecast():
    region = request.args.get('region')
    months = request.args.get('months', default=12, type=int)

    if not region:
        return jsonify({'error': 'Please provide a region using ?region=YourRegion'}), 400

    # Filter data and forecast
    try:
        forecast_df = generate_forecast(df, region, months)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Format for JSON
    forecast_df['ds'] = forecast_df['ds'].astype(str)

    response = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
