import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays
import traceback

st.set_page_config(page_title="Forecasting Final", layout="wide")
st.title("📊 Previsão de Vendas (14 Dias) - Versão Final")

# --- 1. FUNÇÕES AUXILIARES ---
def get_holidays_calendar(start_date, end_date):
    try:
        br_holidays = holidays.Brazil(subdiv='SP', state='SP')
        date_range = pd.date_range(start_date, end_date)
        return pd.DataFrame([
            {'Date': d, 'IsHoliday': 1 if d in br_holidays else 0} 
            for d in date_range
        ])
    except:
        return pd.DataFrame({'Date': pd.date_range(start_date, end_date), 'IsHoliday': 0})

def get_live_forecast(days=14, lat=-23.55, lon=-46.63):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, 
            "longitude": lon, 
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "America/Sao_Paulo", 
            "forecast_days": days + 2
        }
        r = requests.get(url, params=params).json()
        dates = pd.to_datetime(r['daily']['time'])
        t_avg = (np.array(r['daily']['temperature_2m_max']) + np.array(r['daily']['temperature_2m_min'])) / 2
        return pd.DataFrame({
            'Date': dates, 
            'Temp_Avg': t_avg, 
            'Rain_mm': r['daily']['precipitation_sum']
        })
    except: 
        return None

# --- 2. CARGA E CATEGORIZAÇÃO ---
def classify_group(desc):
    if not isinstance(desc, str): return 'Outros'
    desc = desc.lower()
    
    if 'americana bola' in desc: return 'Americana Bola'
    if any(x in desc for x in ['vero', 'primavera', 'roxa']): return 'Vero'
    if 'mini' in desc: return 'Minis'
    if any(x in desc for x in ['legume', 'cenoura', 'beterraba', 'abobrinha']): return 'Legumes'
    if any(x in desc for x in ['salada', 'alface', 'rúcula', 'agrião']): return 'Saladas'
    return 'Outros'

@st.cache_data
def load_data(uploaded_file):
    try:
        try: df = pd.read_csv(uploaded_file, sep=',') 
        except: df = pd.read_excel(uploaded_file)
        if df.shape[1] < 2: df = pd.read_csv(uploaded_file, sep=';')
        
        df.columns = df.columns.str.strip()
        
        rename_map = {
            'Data':'Date','Dia':'Date','Cod- SKU':'SKU','Código':'SKU',
            'Produto.DS_PRODUTO':'Description','Descrição':'Description',
            'Qtde':'Orders','Pedidos':'Orders'
        }
        df = df.rename(columns=rename_map)
        
        if 'Description' not in df.columns: 
            if len(df.columns) >= 4: 
                new_cols = ['Date','SKU','Description','Orders'] + list(df.columns[4:])
                df.columns = new_cols
            else: 
                df['Description'] = 'Prod ' + df['SKU'].astype(str)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
        df['Group'] = df['Description'].apply(classify_group)
        
        return df.groupby(['Date','SKU','Description','Group'])['Orders'].sum().reset_index()
    except Exception as e:
        st.error(f"Erro leitura: {e}")
        return pd.DataFrame()

# --- 3. PRÉ-PROCESSAMENTO ---
def filter_history_vero(df):
    try:
        mask_vero = df['Group'] == 'Vero'
        mask_date = df['Date'] >= '2025-01-01'
        keep_vero = mask_vero & mask_date
        keep_others = ~mask_vero
        return df[keep_vero | keep_others].copy()
    except:
        return df

def clean_outliers(df):
    try:
        df = df.sort_values(['SKU', 'Date'])
        target_groups = ['Vero', 'Americana Bola']
        skus = df[df['Group'].isin(target_groups)]['SKU'].unique()
        
        for sku in skus:
            mask = df['SKU'] == sku
            series = df.loc[mask, 'Orders']
            if len(series) < 5: continue
            
            roll_med = series.rolling(window=14, min_periods=1, center=True).median()
            is_outlier = series > (roll_med * 4)
            if is_outlier.any():
                df.loc[mask & is_outlier, 'Orders'] = roll_med[is_outlier]
        return df
    except:
        return df

# --- 4. MOTOR DE PREVISÃO ---
def run_forecast(df_hist_raw, days_ahead=14):
    # Removido o try/except externo para evitar erro de indentação
    
    # 1. Filtros
    df_hist = filter_history_vero(df_hist_raw)
    df_hist = clean_outliers(df_hist)
    
    end_date = df_hist['Date'].max()
    start_date = df_hist['Date'].min()
    dates_future = pd.date_range(start_date, end_date + timedelta(days=days_ahead))
    
    # 2. Base Clima e Feriados
    df_dates = pd.DataFrame({'Date': dates_future})
    weather = get_live_forecast(days=days_ahead)
    
    np.random.seed(42)
    df_dates['Temp_Avg'] = np.random.normal(25, 3, len(df_dates))
    is_summer = df_dates['Date'].dt.month.isin([1,2,3,12])
    df_dates['Rain_mm'] = np.where(is_summer, np.random.exponential(8, len(df_dates)), 4)
    
    if weather is not None:
        weather['Date'] = pd.to_datetime(weather['Date'])
        df_dates = pd.merge(df_dates, weather, on='Date', how='left', suffixes=('', '_real'))
        df_dates['Temp_Avg'] = df_dates['Temp_Avg_real'].fillna(df_dates['Temp_Avg'])
        df_dates['Rain_mm'] = df_dates['Rain_mm_real'].fillna(df_dates['Rain_mm'])
        df_dates = df_