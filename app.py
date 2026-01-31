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
st.title("📊 Previsão de Vendas (14 Dias) - Consolidado")

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
        # Fallback seguro
        d_range = pd.date_range(start_date, end_date)
        return pd.DataFrame({'Date': d_range, 'IsHoliday': 0})

def get_live_forecast(days=14, lat=-23.55, lon=-46.63):
    try:
        # Busca dados de clima
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
    desc_lower = desc.lower()
    
    # 1. Prioridade: Americana Bola
    if 'americana bola' in desc_lower: 
        return 'Americana Bola'
    
    # 2. Vero / Primavera / Roxa
    if any(x in desc_lower for x in ['vero', 'primavera', 'roxa']): 
        return 'Vero'
    
    # 3. Minis
    if 'mini' in desc_lower: 
        return 'Minis'
    
    # 4. REGRA INSALATA: > 100g vai para Saladas
    if 'insalata' in desc_lower:
        match = re.search(r'(\d+)\s*g', desc_lower)
        if match:
            weight = int(match.group(1))
            if weight > 100:
                return 'Saladas'
    
    # 5. Legumes
    legumes_list = ['legume', 'cenoura', 'beterraba', 'abobrinha']
    if any(x in desc_lower for x in legumes_list): 
        return 'Legumes'
    
    # 6. Saladas (Geral)
    saladas_list = ['salada', 'alface', 'rúcula', 'agrião', 'insalata']
    if any(x in desc_lower for x in saladas_list): 
        return 'Saladas'
    
    return 'Outros'

@st.cache_data
def load_data(uploaded_file):
    try:
        try: df = pd.read_csv(uploaded_file, sep=',') 
        except: df = pd.read_excel(uploaded_file)
        if df.shape[1] < 2: df = pd.read_csv(uploaded_file, sep=';')
        
        df.columns = df.columns.str.strip()
        
        rename_map = {
            'Data':'Date','Dia':'Date',
            'Cod- SKU':'SKU','Código':'SKU',
            'Produto.DS_PRODUTO':'Description',
            'Descrição':'Description',
            'Qtde':'Orders','Pedidos':'Orders'
        }
        df = df.rename(columns=rename_map)
        
        # Garante coluna Description
        if 'Description' not in df.columns: 
            if len(df.columns) >= 4: 
                # Assume coluna 3 como descrição
                cols = ['Date','SKU','Description','Orders']
                df.columns = cols + list(df.columns[4:])
            else: 
                df['Description'] = 'Prod ' + df['SKU'].astype(str)

        # Tratamento de Tipos
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
        
        # --- APLICA CLASSIFICAÇÃO ---
        df['Group'] = df['Description'].apply(classify_group)
        
        # Agrupa para remover duplicatas
        return df.groupby(['Date','SKU','Description','Group'])['Orders'].sum().reset_index()
    except Exception as e:
        st.error(f"Erro na leitura do arquivo: {e}")
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
        targets = ['Vero', 'Americana Bola']
        skus = df[df['Group'].isin(targets)]['SKU'].unique()
        
        for sku in skus:
            mask = df['SKU'] == sku
            series = df.loc[mask, 'Orders']
            if len(series) < 5: continue
            
            # Limpeza estatística
            median_val = series.rolling(window=14, min_periods=1, center=True).median()
            is_outlier = series > (median_val * 4)
            
            if is_outlier.any():
                df.loc[mask &