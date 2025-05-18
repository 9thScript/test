from prophet import Prophet
import pandas as pd

def generate_forecast(df, region, months):
    # Filter data for selected region
    region_df = df[df['Region'] == region].copy()
    region_df['ds'] = pd.to_datetime(region_df['Year'].astype(str) + '-' + region_df['Month'].astype(str).str.zfill(2) + '-01')
    region_df['y'] = region_df['Dengue_Cases']
    region_df = region_df[['ds', 'y']].dropna()

    # Train Prophet
    m = Prophet(interval_width=0.95, daily_seasonality=False)
    m.fit(region_df)

    # Create future dates
    future = m.make_future_dataframe(periods=months, freq='MS')
    forecast = m.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Merge forecast with actual data
    merged = pd.merge(forecast, region_df, on='ds', how='left')

    # Label whether the row has actual data or is forecast
    merged['type'] = merged['y'].apply(lambda x: 'actual' if pd.notnull(x) else 'forecast')

    return merged
