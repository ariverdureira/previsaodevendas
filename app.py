import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays
import traceback
import re

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
    desc_lower = desc.lower()
    
    # 1. Prioridade Absoluta
    if 'americana bola' in desc_lower: return 'Americana Bola'
    
    # 2. Vero
    if any(x in desc_lower for x in ['vero', 'primavera', 'roxa']): return 'Vero'
    
    # 3. Minis
    if 'mini' in desc_lower: return 'Minis'
    
    # 4. REGRA ESPECIAL: Insalata > 100g vai para Saladas
    if 'insalata' in desc_lower:
        match = re.search(r'(\d+)\s*g', desc_lower)
        if match:
            weight = int(match.group(1))
            if weight > 100:
                return 'Saladas'
    
    # 5. Legumes
    if any(x in desc_lower for x in ['legume', 'cenoura', 'beterraba', 'abobrinha']): return 'Legumes'
    
    # 6. Saladas (Regra Geral)
    if any(x in desc_lower for x in ['salada', 'alface', 'rúcula', 'agrião', 'insalata']): return 'Saladas'
    
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
        df['Group'] = df['Description'].apply(classify