import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays
import traceback
import re

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Forecasting Final", layout="wide")
st.title("📊 Previsão de Vendas (14 Dias) - Consolidado")

# --- 1. FUNÇÕES AUXILIARES ---

def get_holidays_calendar(start_date, end_date):
    try:
        br_holidays = holidays.Brazil(subdiv='SP', state='SP')
        date_range = pd.date_range(start_date, end_date)
        data = []
        for d in date_range:
            is_hol = 1 if d in br_holidays else 0
            data.append({'Date': d, 'IsHoliday': is_hol})
        return pd.DataFrame(data)
    except:
        d_range = pd.date_range(start_date, end_date)
        return pd.DataFrame({'Date': d_range, 'IsHoliday': 0})

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
        t_max = np.array(r['daily']['temperature_2m_max'])
        t_min = np.array(r['daily']['temperature_2m_min'])
        t_avg = (t_max + t_min) / 2
        
        return pd.DataFrame({
            'Date': dates, 
            'Temp_Avg': t_avg, 
            'Rain_mm': r['daily']['precipitation_sum']
        })
    except: 
        return None

# --- 2. CLASSIFICAÇÃO DE GRUPOS ---

def classify_group(desc):
    if not isinstance(desc, str): return 'Outros'
    txt = desc.lower()
    
    if 'americana bola' in txt: return 'Americana Bola'
    if any(x in txt for x in ['vero', 'primavera', 'roxa']): return 'Vero'
    if 'mini' in txt: return 'Minis'
    
    if 'insalata' in txt:
        match = re.search(r'(\d+)\s*g', txt)
        if match:
            weight = int(match.group(1))
            if weight > 100: return 'Saladas'
    
    legumes = ['legume', 'cenoura', 'beterraba', 'abobrinha']
    if any(x in txt for x in legumes): return 'Legumes'
    
    saladas = ['salada', 'alface', 'rúcula', 'agrião', 'insalata']
    if any(x in txt for x in saladas): return 'Saladas'
    
    return 'Outros'

@st.cache_data
def load_data(uploaded_file):
    try:
        try: df = pd.read_csv(uploaded_file, sep=',') 
        except: df = pd.read_excel(uploaded_file)
        
        if df.shape[1] < 2: df = pd.read_csv(uploaded_file, sep=';')
        
        df.columns = df.columns.str.strip()
        
        rename_map = {
            'Data':'Date', 'Dia':'Date',
            'Cod- SKU':'SKU', 'Código':'SKU',
            'Produto.DS_PRODUTO':'Description', 'Descrição':'Description',
            'Pedidos':'Orders', 'Qtde':'Orders'
        }
        df = df.rename(columns=rename_map)
        
        if 'Description' not in df.columns:
            if len(df.columns) >= 4:
                cols = ['Date','SKU','Description','Orders']
                existing = list(df.columns)
                df.columns = cols + existing[4:]
            else:
                df['Description'] = 'Prod ' + df['SKU'].astype(str)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
        df['Group'] = df['Description'].apply(classify_group)
        
        return df.groupby(['Date','SKU','Description','Group'])['Orders'].sum().reset_index()
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return pd.DataFrame()

# --- 3. PRÉ-PROCESSAMENTO PARA IA ---

def filter_history_vero(df):
    # Filtra histórico APENAS para o treino da IA
    mask_vero = df['Group'] == 'Vero'
    mask_date = df['Date'] >= '2025-01-01'
    keep = (mask_vero & mask_date) | (~mask_vero)
    return df[keep].copy()

def clean_outliers(df):
    df = df.sort_values(['SKU', 'Date'])
    targets = ['Vero', 'Americana Bola']
    skus_to_check = df[df['Group'].isin(targets)]['SKU'].unique()
    
    for sku in skus_to_check:
        mask = df['SKU'] == sku
        series = df.loc[mask, 'Orders']
        if len(series) < 5: continue
        
        roll_med = series.rolling(14, min_periods=1, center=True).median()
        is_outlier = series > (roll_med * 4)
        
        if is_outlier.any():
            df.loc[mask & is_outlier, 'Orders'] = roll_med[is_outlier]
    return df

def generate_features(df):
    d = df.sort_values(['SKU', 'Date']).copy()
    d['DayOfWeek'] = d['Date'].dt.dayofweek
    d['IsWeekend'] = (d['DayOfWeek'] >= 5).astype(int)
    
    d['lag_1'] = d.groupby('SKU')['Orders'].shift(1)
    d['lag_7'] = d.groupby('SKU')['Orders'].shift(7)
    d['lag_14'] = d.groupby('SKU')['Orders'].shift(14)
    d['roll_7'] = d.groupby('SKU')['Orders'].shift(1).rolling(7).mean()
    return d

# --- 4. MOTOR DE PREVISÃO ---

def run_forecast(df_raw, days_ahead=14):
    # 1. Filtros (Somente para a IA)
    df_train_base = filter_history_vero(df_raw)
    df_train_base = clean_outliers