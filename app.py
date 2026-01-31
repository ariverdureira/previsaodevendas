import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays
import traceback # Para mostrar o erro real se acontecer

st.set_page_config(page_title="Forecasting Diagnóstico", layout="wide")
st.title("📊 Previsão de Vendas (14 Dias) - Modo Seguro")

# --- 1. FUNÇÕES AUXILIARES ---
def get_holidays_calendar(start_date, end_date):
    try:
        br_holidays = holidays.Brazil(subdiv='SP', state='SP')
        date_range = pd.date_range(start_date, end_date)
        return pd.DataFrame([{'Date': d, 'IsHoliday': 1 if d in br_holidays else 0} for d in date_range])
    except Exception as e:
        st.warning(f"Erro ao carregar feriados: {e}. Seguindo sem feriados.")
        return pd.DataFrame({'Date': pd.date_range(start_date, end_date), 'IsHoliday': 0})

def get_live_forecast(days=14, lat=-23.55, lon=-46.63):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                  "timezone": "America/Sao_Paulo", "forecast_days": days + 2}
        r = requests.get(url, params=params).json()
        dates = pd.to_datetime(r['daily']['time'])
        t_avg = (np.array(r['daily']['temperature_2m_max']) + np.array(r['daily']['temperature_2m_min'])) / 2
        return pd.DataFrame({'Date': dates, 'Temp_Avg': t_avg, 'Rain_mm': r['daily']['precipitation_sum']})
    except: 
        return None

# --- 2. CARGA E CATEGORIZAÇÃO ---
def classify_group(desc):
    if not isinstance(desc, str):
        return 'Outros'
    desc = desc.lower()
    
    # PRIORIDADE 1: Americana Bola
    if 'americana bola' in desc: return 'Americana Bola'
    
    # Demais Grupos
    if any(x in desc for x in ['vero', 'primavera', 'roxa']): return 'Vero'
    if 'mini' in desc: return 'Minis'
    if any(x in desc for x in ['legume', 'cenoura', 'beterraba', 'abobrinha', 'vagem', 'tomate']): return 'Legumes'
    if any(x in desc for x in ['salada', 'alface', 'rúcula', 'agrião', 'espinafre', 'couve']): return 'Saladas'
    
    return 'Outros'

@st.cache_data
def load_data(uploaded_file):
    try:
        try: df = pd.read_csv(uploaded_file, sep=',') 
        except: df = pd.read_excel(uploaded_file)
        if df.shape[1] < 2: df = pd.read_csv(uploaded_file, sep=';')
        
        df.columns = df.columns.str.strip()
        
        # Mapeamento flexível de colunas
        rename = {'Data':'Date','Dia':'Date','Cod- SKU':'SKU','Código':'SKU',
                  'Produto.DS_PRODUTO':'Description','Descrição':'Description',
                  'Qtde':'Orders','Pedidos':'Orders'}
        df = df.rename(columns=rename)
        
        # Garante que Description existe
        if 'Description' not in df.columns: 
            if len(df.columns) >= 4: 
                df.columns = ['Date','SKU','Description','Orders'] + list(df.columns[4:])
            else: 
                df['Description'] = 'Prod ' + df['SKU'].astype(str)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
        
        # Aplica Categorização Nova
        df['Group'] = df['Description'].apply(classify_group)
        
        return df.groupby(['Date','SKU','Description','Group'])['Orders'].sum().reset_index()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return pd.DataFrame()

# --- 3. PRÉ-PROCESSAMENTO ---
def filter_history_vero(df):
    try:
        vero_mask = df['Group'] == 'Vero'
        # Filtra Vero apenas a partir de 2025
        keep_vero = vero_mask & (df['Date'] >= '2025-01-01')
        keep_others = ~vero_mask
        return df[keep_vero | keep_others].copy()
    except:
        return df

def clean_outliers(df):
    try:
        df = df.sort_values(['SKU', 'Date'])
        # Aplica limpeza no grupo Vero e também no Americana Bola
        target_groups = ['Vero', 'Americana Bola']
        target_skus = df[df['Group'].isin(target_groups)]['SKU'].unique()
        
        for sku in target_skus:
            mask = df['SKU'] == sku
            series = df.loc[mask, 'Orders']
            # Se tiver poucos dados, ignora
            if len(series) < 5: continue
            
            rolling_median = series.rolling(window=14, min_periods=1, center=True).median()
            is_outlier = series > (rolling_median * 4) 
            if is_outlier.any():
                df.loc[mask & is_outlier, 'Orders'] = rolling_median[is_outlier]
        return df
    except Exception as e:
        st.warning(f"Aviso: Não foi possível limpar outliers ({e}). Seguindo com dados brutos.")
        return df

# --- 4. MOTOR DE PREVISÃO ---
def run_forecast(df_hist_raw, days_ahead=14):
    try:
        # Etapas de limpeza com status
        status = st.empty()
        status.text("Processando: Filtrando histórico...")
        df_hist = filter_history_vero(df_hist_raw)
        
        status.text("Processando: Limpando picos...")
        df_hist = clean_outliers(df_hist)
        
        end_date_hist = df_hist['Date'].max()
        dates_future = pd.date_range(df_hist['Date'].min(), end_date_hist + timedelta(days=days_ahead))
        
        # Base
        df_dates = pd.DataFrame({'Date': dates_future})
        
        # Clima
        status.text("Processando: Buscando clima...")
        real_forecast = get_live_forecast(days=days_ahead)
        np.random.seed(42)
        df_dates['Temp_Avg'] = np.random.normal(25, 3, len(df_dates))
        df_dates['Rain_mm'] = np.where(df_dates['Date'].dt.month.isin([1,2,3,12]), np.random.exponential(8, len(df_dates)), 4)
        
        if real_forecast is not None:
            real_forecast['Date'] = pd.to_datetime(real_forecast['Date'])
            df_dates = pd.merge(df_dates, real_forecast, on='Date', how='left', suffixes=('', '_real'))
            df_dates['Temp_Avg'] = df_dates['Temp_Avg_real'].fillna(df_dates['Temp_Avg'])
            df_dates['Rain_mm'] = df_dates['Rain_mm_real'].fillna(df_dates['Rain_mm'])
            df_dates = df_dates[['Date', 'Temp_Avg', 'Rain_mm']]

        # Feriados
        status.text("Processando: Calculando feriados...")
        df_holidays = get_holidays_calendar(df_dates['Date'].min(), df_dates['Date'].max())
        df_dates = pd.merge(df_dates, df_holidays, on='Date', how='left')
        df_dates['IsHoliday'] = df_dates['IsHoliday'].fillna(0)

        # Merge Full
        unique_skus = df_hist[['SKU', 'Description', 'Group']].drop_duplicates()
        unique_skus['key'] = 1; df_dates['key'] = 1
        df_master = pd.merge(df_dates, unique_skus, on='key').drop('key', axis=1)
        df_master = pd.merge(df_master, df_hist[['Date','SKU','Orders']], on=['Date','SKU'], how='left')
        
        # Preenche zeros no histórico
        mask_hist = df_master['Date'] <= end_date_hist
        df_master.loc[mask_hist, 'Orders'] = df_master.loc[mask_hist, 'Orders'].fillna(0)

        # Features
        status.text("Processando: Engenharia de atributos (XGBoost)...")
        def gen_feat(d):
            d = d.sort_values(['SKU', 'Date'])
            d['DayOfWeek'] = d['Date'].dt.dayofweek
            d['IsWeekend'] = (d['DayOfWeek'] >= 5).astype(int)
            for l in [1,7,14]: 
                d[f'lag_{l}'] = d.groupby('SKU')['Orders'].shift(l)
            d['rolling_mean_7'] = d.groupby('SKU')['Orders'].shift(1).rolling(7).mean()
            return d

        df_feat = gen_feat(df_master)
        train = df_feat[df_feat['Date'] <= end_date_hist].dropna()
        
        if train.empty:
            st.error("Erro Crítico: Não há dados suficientes para treinar o modelo após os filtros. Verifique se há histórico > 14 dias.")
            return pd.DataFrame(), df_hist_raw

        # Treino
        status.text("Processando: Treinando Inteligência Artificial...")
        feat_cols = ['DayOfWeek','IsWeekend','IsHoliday','Temp_Avg','Rain_mm','lag_1','lag_7','rolling_mean_7']
        model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, n_jobs=-1)
        model.fit(train[feat_cols], train['Orders'])

        # Loop Previsão
        preds = []
        curr = df_master[df_master['Date'] <= end_date_hist].copy()
        prog = st.progress(0)
        
        status.text("Processando: Gerando previsão futura...")
        for i in range(1, days_ahead + 1):
            nxt = end_date_hist + timedelta(days=i)
            
            cols_base = ['Date','SKU','Description','Group','Temp_Avg','Rain_mm','IsHoliday']
            nxt_base = df_master[df_master['Date'] == nxt][cols_base].copy()
            
            tmp = pd.concat([curr, nxt_base], sort=False)
            tmp = gen_feat(tmp)
            row = tmp[tmp['Date'] == nxt].copy()
            
            # Predição
            y = np.maximum(model.predict(row[feat_cols].fillna(0)), 0)
            row['Orders'] = y.round(0)
            
            # --- TRAVA DE NEGÓCIO ---
            is_holiday_flag = row['IsHoliday'].values[0] == 1
            is_sunday = nxt.dayofweek == 6
            
            if is_sunday or is_holiday_flag:
                row['Orders'] = 0
                
            preds.append(row)
            curr = pd.concat([curr, row],