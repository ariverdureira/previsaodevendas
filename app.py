import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import plotly.express as px
import requests  # Biblioteca para conectar na API de Clima

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Forecasting Vero + Clima Real", layout="wide")

st.title("🌦️ Previsão de Vendas Inteligente (Vero & Mix)")
st.markdown("Modelo XGBoost alimentado com **Dados Históricos** + **Previsão do Tempo em Tempo Real** (Open-Meteo).")

# --- 1. FUNÇÃO DE CLIMA (API ONLINE) ---
def get_live_forecast(days=11, lat=-23.55, lon=-46.63):
    """
    Busca a previsão do tempo real para São Paulo (ou lat/lon configurada).
    Retorna um DataFrame com Data, Temp_Avg e Rain_mm.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "America/Sao_Paulo",
            "forecast_days": days + 2 # Pega uma margem
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Processar JSON
        daily_data = data['daily']
        dates = pd.to_datetime(daily_data['time'])
        temp_max = np.array(daily_data['temperature_2m_max'])
        temp_min = np.array(daily_data['temperature_2m_min'])
        rain = np.array(daily_data['precipitation_sum'])
        
        # Calcular Média
        temp_avg = (temp_max + temp_min) / 2
        
        df_weather = pd.DataFrame({
            'Date': dates,
            'Temp_Avg': temp_avg,
            'Rain_mm': rain
        })
        return df_weather
        
    except Exception as e:
        st.warning(f"⚠️ Não foi possível buscar o clima online. Usando médias históricas. Erro: {e}")
        return None

# --- 2. FUNÇÕES DO MOTOR DE DADOS ---
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=',')
        if df.shape[1] < 2:
            df = pd.read_csv(uploaded_file, sep=';')
    except:
        df = pd.read_excel(uploaded_file)
        
    cols = df.columns
    # Padronização de colunas
    if 'Data' in cols and 'Cod- SKU' in cols:
        df.rename(columns={'Data': 'Date', 'Cod- SKU': 'SKU', 'Pedidos': 'Orders'}, inplace=True)
    else:
        df.columns = ['Date', 'SKU', 'Description', 'Orders'] + list(cols[4:])
        df = df[['Date', 'SKU', 'Description', 'Orders']]
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
    
    return df.groupby(['Date', 'SKU', 'Description'])['Orders'].sum().reset_index()

def generate_features(df):
    df = df.sort_values(['SKU', 'Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    for lag in [1, 7, 14]:
        df[f'lag_{lag}'] = df.groupby('SKU')['Orders'].shift(lag)
    df['rolling_mean_7'] = df.groupby('SKU')['Orders'].shift(1).rolling(7).mean()
    
    return df

def run_forecast(df_hist, days_ahead=11):
    end_date_hist = df_hist['Date'].max()
    dates_future = pd.date_range(start=df_hist['Date'].min(), end=end_date_hist + timedelta(days=days_ahead))
    
    # 1. SETUP CLIMÁTICO (HISTÓRICO SIMULADO + FUTURO REAL)
    
    # A) Busca clima Futuro Real (API)
    real_forecast = get_live_forecast(days=days_ahead)
    
    # B) Prepara DataFrame de Datas
    df_dates = pd.DataFrame({'Date': dates_future})
    
    # C) Simula Histórico (para o treino passado)
    np.random.seed(42)
    # Gera clima base para tudo
    df_dates['Temp_Avg'] = np.random.normal(25, 3, len(df_dates))
    df_dates['Rain_mm'] = np.where(df_dates['Date'].dt.month.isin([1,2,3,12]), 
                                   np.random.exponential(8, len(df_dates)), 
                                   np.random.exponential(4, len(df_dates)))
    
    # D) Sobrescreve o Futuro com o Clima Real (API)
    if real_forecast is not None:
        # Merge para atualizar apenas as datas que temos previsão real
        # Primeiro garantimos que as datas estão no mesmo formato
        real_forecast['Date'] = pd.to_datetime(real_forecast['Date'])
        
        # Atualiza Temp e Chuva onde a data bate
        for idx, row in real_forecast.iterrows():
            mask = df_dates['Date'] == row['Date']
            if mask.any():
                df_dates.loc[mask, 'Temp_Avg'] = row['Temp_Avg']
                df_dates.loc[mask, 'Rain_mm'] = row['Rain_mm']
    
    # 2. MERGE COM SKUS
    unique_skus = df_hist[['SKU', 'Description']].drop_duplicates()
    unique_skus['key'] = 1
    df_dates['key'] = 1
    
    df_master = pd.merge(df_dates, unique_skus, on='key').drop('key', axis=1)
    df_master = pd.merge(df_master, df_hist[['Date', 'SKU', 'Orders']], on=['Date', 'SKU'], how='left')
    
    mask_hist = df_master['Date'] <= end_date_hist
    df_master.loc[mask_hist, 'Orders'] = df_master.loc[mask_hist, 'Orders'].fillna(0)
    
    # 3. TREINAMENTO XGBOOST
    df_feat = generate_features(df_master)
    train = df_feat[df_feat['Date'] <= end_date_hist].dropna()
    
    features = ['DayOfWeek', 'IsWeekend', 'Temp_Avg', 'Rain_mm', 'lag_1', 'lag_7', 'rolling_mean_7']
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1)
    model.fit(train[features], train['Orders'])
    
    # 4. REGRA DE NEGÓCIO: VERO OVERRIDE (NOVO CLIENTE)
    # Pega média dos últimos 3 dias REAIS como base
    last_3_days = end_date_hist - timedelta(days=3)
    vero_data = df_hist[(df_hist['Date'] > last_3_days) & 
                        (df_hist['Description'].str.lower().str.contains('vero'))]
    vero_means = vero_data.groupby('SKU')['Orders'].mean()
    
    # 5. LOOP DE PREVISÃO
    future_preds = []
    current_hist = df_master[df_master['Date'] <= end_date_hist].copy()
    
    prog_bar = st.progress(0)
    
    for i in range(1, days_ahead + 1):
        next_date = end_date_hist + timedelta(days=i)
        
        # Dados do dia (já com clima real)
        base_cols = ['Date', 'SKU', 'Description', 'Temp_Avg', 'Rain_mm']
        next_day = df_master[df_master['Date'] == next_date][base_cols].copy()
        
        # Features
        temp = pd.concat([current_hist, next_day], sort=False)
        temp = generate_features(temp)
        row_pred = temp[temp['Date'] == next_date].copy()
        
        # Predição
        X = row_pred[features].fillna(0)
        y = model.predict(X)
        row_pred['Orders'] = np.maximum(y, 0)
        
        # Aplica Override Vero (com ajuste de fds)
        for sku, mean_val in vero_means.items():
            mask = row_pred['SKU'] == sku
            is_weekend = row_pred.loc[mask, 'IsWeekend'] == 1
            final_val = mean_val * (0.8 if is_weekend.any() else 1.0)
            row_pred.loc[mask, 'Orders'] = final_val
            
        row_pred['Orders'] = row_pred['Orders'].round(0)
        
        future_preds.append(row_pred)
        current_hist = pd.concat([current_hist, row_pred], sort=False)
        prog_bar.progress(i / days_ahead)
        
    return pd.concat(future_preds), real_forecast

# --- 3. INTERFACE ---
uploaded_file = st.file_uploader("📂 Carregue o arquivo data.xlsx", type=['csv', 'xlsx'])

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    last_date = df_raw['Date'].max().date()
    
    st.info(f"Base carregada até: **{last_date}**. Pronto para buscar previsão do tempo e gerar vendas.")
    
    if st.button("🚀 Gerar Previsão com Clima Real"):
        with st.spinner('Consultando API de Clima e Processando XGBoost...'):
            forecast, weather_used = run_forecast(df_raw)
            
            # Exibir Clima Usado
            if weather_used is not None:
                st.subheader("🌦️ Clima Considerado (Próximos Dias - SP)")
                # Gráfico simples de clima
                col_w1, col_w2 = st.columns([3, 1])
                fig_w = px.bar(weather_used, x='Date', y='Temp_Avg', title="Temperatura Média Prevista (°C)")
                fig_w.add_bar(x=weather_used['Date'], y=weather_used['Rain_mm'], name='Chuva (mm)')
                col_w1.plotly_chart(fig_w, use_container_width=True)
                col_w2.dataframe(weather_used[['Date', 'Temp_Avg', 'Rain_mm']].round(1), height=300)
            
            # KPI Cards
            total_vol = int(forecast['Orders'].sum())
            vero_vol = int(forecast[forecast['Description'].str.contains('VERO')]['Orders'].sum())
            
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("Volume Total (11 dias)", f"{total_vol:,}")
            col2.metric("Volume Linha Vero", f"{vero_vol:,}")
            
            # Gráfico Vendas
            st.subheader("📈 Previsão de Vendas Detalhada")
            skus = forecast['Description'].unique()
            sel_sku = st.selectbox("Filtrar Produto:", skus)
            
            fig = px.line(forecast[forecast['Description'] == sel_sku], 
                          x='Date', y='Orders', title=f"Demanda Prevista: {sel_sku}", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Planilha Final", csv, "previsao_com_clima.csv", "text/csv")