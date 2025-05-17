from flask import Flask, jsonify
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
import joblib

app = Flask(__name__)

# Load trained model and cleaned data
model = joblib.load('prophet_model.pkl')
df = pd.read_csv('cleaned_ncr_dengue.csv')
df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime format

@app.route('/')
def home():
    return "<h2>Dengue Forecast API</h2><p>Visit <code>/forecast</code> for data or <code>/plot</code> for the graph.</p>"

@app.route('/forecast', methods=['GET'])
def forecast():
    # Forecast for the next 12 months (you can adjust this)
    future = model.make_future_dataframe(periods=12, freq='MS')  # Month Start
    forecast = model.predict(future)

    # Resample monthly forecast
    forecast_monthly = forecast.resample('MS', on='ds').mean().reset_index()

    # Prepare data for JSON response
    response = forecast_monthly[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    response['ds'] = response['ds'].astype(str)  # Convert to string for JSON
    return jsonify(response.to_dict(orient='records'))

@app.route('/plot', methods=['GET'])
def plot():
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)
    forecast_monthly = forecast.resample('MS', on='ds').mean().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_monthly['ds'],
        y=forecast_monthly['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_monthly['ds'].tolist() + forecast_monthly['ds'][::-1].tolist(),
        y=forecast_monthly['yhat_upper'].tolist() + forecast_monthly['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(173,216,230,0.4)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='Confidence Interval'
    ))

    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Actual',
        line=dict(color='black', dash='dot')
    ))

    fig.update_layout(
        title='NCR Dengue Case Forecast',
        xaxis_title='Date',
        yaxis_title='Dengue Cases',
        hovermode='x',
        template='plotly_white'
    )

    # Convert Plotly figure to HTML and return
    return fig.to_html()

if __name__ == '__main__':
    app.run(debug=True)
