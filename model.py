from prophet import Prophet
import pandas as pd

def generate_forecast(df, region, months):
    region_df = df[df['Region'] == region].copy()
    region_df['ds'] = pd.to_datetime(region_df['Year'].astype(str) + '-' + region_df['Month'].astype(str).str.zfill(2) + '-01')
    region_df['y'] = region_df['Dengue_Cases']
    region_df = region_df[['ds', 'y']].dropna()

    m = Prophet(interval_width=0.95, daily_seasonality=False)
    m.fit(region_df)

    future = m.make_future_dataframe(periods=months, freq='MS')
    forecast = m.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months)