import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import plotly.express as px
import requests

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Forecasting Sazonal Vero", layout="wide")
st.title("🌦️ Previsão de Vendas - Sazonalidade Ajustada")

# ... (Mantenha as funções get_live_forecast, load_data e generate_features iguais ao anterior) ...
# Vou colocar aqui apenas a função run_forecast alterada, que é onde a mágica acontece.

def run_forecast(df_hist, days_ahead=11):
    end_date_hist = df_hist['Date'].max()
    dates_future = pd.date_range(start=df_hist['Date'].min(), end=end_date_hist + timedelta(days=days_ahead))
    
    # 1. Clima
    real_forecast = get_live_forecast(days=days_ahead)
    df_dates = pd.DataFrame({'Date': dates_future})
    
    np.random.seed(42)
    df_dates['Temp_Avg'] = np.random.normal(25, 3, len(df_dates))
    df_dates['Rain_mm'] = np.where(df_dates['Date'].dt.month.isin([1,2,3,12]), 
                                   np.random.exponential(8, len(df_dates)), 
                                   np.random.exponential(4, len(df_dates)))
    
    if real_forecast is not None:
        real_forecast['Date'] = pd.to_datetime(real_forecast['Date'])
        for idx, row in real_forecast.iterrows():
            mask = df_dates['Date'] == row['Date']
            if mask.any():
                df_dates.loc[mask, 'Temp_Avg'] = row['Temp_Avg']
                df_dates.loc[mask, 'Rain_mm'] = row['Rain_mm']
    
    # 2. Merge Base
    unique_skus = df_hist[['SKU', 'Description']].drop_duplicates()
    unique_skus['key'] = 1
    df_dates['key'] = 1
    
    df_master = pd.merge(df_dates, unique_skus, on='key').drop('key', axis=1)
    df_master = pd.merge(df_master, df_hist[['Date', 'SKU', 'Orders']], on=['Date', 'SKU'], how='left')
    
    mask_hist = df_master['Date'] <= end_date_hist
    df_master.loc[mask_hist, 'Orders'] = df_master.loc[mask_hist, 'Orders'].fillna(0)
    
    # 3. Treino
    df_feat = generate_features(df_master)
    train = df_feat[df_feat['Date'] <= end_date_hist].dropna()
    
    features = ['DayOfWeek', 'IsWeekend', 'Temp_Avg', 'Rain_mm', 'lag_1', 'lag_7', 'rolling_mean_7']
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1)
    model.fit(train[features], train['Orders'])
    
    # 4. CÁLCULO DO LIFT (Fator de Crescimento)
    # Identifica Vero
    vero_mask = df_hist['Description'].str.lower().str.contains('vero|primavera|roxa', regex=True)
    last_3_days = end_date_hist - timedelta(days=3)
    
    # Volume Alvo (Novo Normal)
    vero_target = df_hist[(df_hist['Date'] > last_3_days) & vero_mask].groupby('SKU')['Orders'].mean()
    
    # Volume Base (O que o modelo achava que era normal antes do cliente)
    # Pegamos a média móvel do último dia histórico como proxy do "antigo normal"
    last_rolling = train[train['Date'] == end_date_hist].set_index('SKU')['rolling_mean_7']
    
    lift_factors = {}
    for sku, target in vero_target.items():
        if sku in last_rolling.index:
            base = last_rolling[sku]
            # Se a base era muito pequena, limita o fator para não explodir (max 10x)
            factor = target / base if base > 10 else 1.0
            lift_factors[sku] = min(factor, 10.0)
        else:
            lift_factors[sku] = 1.0

    # 5. Loop Previsão
    future_preds = []
    current_hist = df_master[df_master['Date'] <= end_date_hist].copy()
    
    prog_bar = st.progress(0)
    
    for i in range(1, days_ahead + 1):
        next_date = end_date_hist + timedelta(days=i)
        
        base_cols = ['Date', 'SKU', 'Description', 'Temp_Avg', 'Rain_mm']
        next_day = df_master[df_master['Date'] == next_date][base_cols].copy()
        
        temp = pd.concat([current_hist, next_day], sort=False)
        temp = generate_features(temp)
        row_pred = temp[temp['Date'] == next_date].copy()
        
        # Previsão Bruta (Tem sazonalidade, mas volume antigo)
        X = row_pred[features].fillna(0)
        y = model.predict(X)
        y = np.maximum(y, 0)
        
        # APLICAÇÃO DO LIFT (Mantém a curva, sobe o volume)
        row_pred['Orders'] = y
        for sku, factor in lift_factors.items():
            mask = row_pred['SKU'] == sku
            # Multiplica a previsão sazonal pelo fator de crescimento
            row_pred.loc[mask, 'Orders'] = row_pred.loc[mask, 'Orders'] * factor
            
        row_pred['Orders'] = row_pred['Orders'].round(0)
        
        future_preds.append(row_pred)
        current_hist = pd.concat([current_hist, row_pred], sort=False)
        prog_bar.progress(i / days_ahead)
        
    return pd.concat(future_preds), real_forecast