import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays

st.set_page_config(page_title="Forecasting Vero 2025+", layout="wide")
st.title("📊 Previsão de Vendas (Foco Vero 2025+)")

# --- 1. FUNÇÕES AUXILIARES ---
def get_holidays_calendar(start_date, end_date):
    br_holidays = holidays.Brazil(subdiv='SP', state='SP')
    date_range = pd.date_range(start_date, end_date)
    return pd.DataFrame([{'Date': d, 'IsHoliday': 1 if d in br_holidays else 0} for d in date_range])

def get_live_forecast(days=14, lat=-23.55, lon=-46.63):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                  "timezone": "America/Sao_Paulo", "forecast_days": days + 2}
        r = requests.get(url, params=params).json()
        dates = pd.to_datetime(r['daily']['time'])
        t_avg = (np.array(r['daily']['temperature_2m_max']) + np.array(r['daily']['temperature_2m_min'])) / 2
        return pd.DataFrame({'Date': dates, 'Temp_Avg': t_avg, 'Rain_mm': r['daily']['precipitation_sum']})
    except: return None

# --- 2. CARGA DE DADOS ---
@st.cache_data
def load_data(uploaded_file):
    try: df = pd.read_csv(uploaded_file, sep=',') 
    except: df = pd.read_excel(uploaded_file)
    if df.shape[1] < 2: df = pd.read_csv(uploaded_file, sep=';')
    
    df.columns = df.columns.str.strip()
    rename = {'Data':'Date','Dia':'Date','Cod- SKU':'SKU','Código':'SKU',
              'Produto.DS_PRODUTO':'Description','Descrição':'Description',
              'Qtde':'Orders','Pedidos':'Orders'}
    df = df.rename(columns=rename)
    
    if 'Description' not in df.columns: 
        if len(df.columns) >= 4: df.columns = ['Date','SKU','Description','Orders'] + list(df.columns[4:])
        else: df['Description'] = 'Prod ' + df['SKU'].astype(str)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
    
    return df.groupby(['Date','SKU','Description'])['Orders'].sum().reset_index()

# --- 3. FILTRO DE HISTÓRICO VERO (NOVO) ---
def filter_history_vero(df):
    """
    Remove dados antigos (antes de 2025) APENAS para a linha Vero/Primavera/Roxa.
    Para os outros produtos, mantém o histórico completo.
    """
    # Identifica linhas Vero
    vero_mask = df['Description'].str.lower().str.contains('vero|primavera|roxa', regex=True)
    
    # Regra 1: Se for Vero, Data deve ser >= 2025
    keep_vero = vero_mask & (df['Date'] >= '2025-01-01')
    
    # Regra 2: Se NÃO for Vero, mantém tudo
    keep_others = ~vero_mask
    
    # Junta as duas partes
    df_filtered = df[keep_vero | keep_others].copy()
    
    return df_filtered

def clean_outliers(df):
    # Limpeza suave de picos extremos para não confundir o treino
    df = df.sort_values(['SKU', 'Date'])
    vero_mask = df['Description'].str.lower().str.contains('vero|primavera|roxa', regex=True)
    vero_skus = df[vero_mask]['SKU'].unique()
    
    for sku in vero_skus:
        mask = df['SKU'] == sku
        series = df.loc[mask, 'Orders']
        rolling_median = series.rolling(window=14, min_periods=1, center=True).median()
        # Tolerância maior (4x) para aceitar picos de venda, mas cortar implantação absurda
        is_outlier = series > (rolling_median * 4) 
        if is_outlier.any():
            df.loc[mask & is_outlier, 'Orders'] = rolling_median[is_outlier]
            
    return df

# --- 4. MOTOR DE PREVISÃO ---
def run_forecast(df_hist_raw, days_ahead=14):
    # APLICAR O FILTRO DE DATA AQUI
    df_hist = filter_history_vero(df_hist_raw)
    
    # Limpeza de outliers
    df_hist = clean_outliers(df_hist)
    
    end_date_hist = df_hist['Date'].max()
    dates_future = pd.date_range(df_hist['Date'].min(), end_date_hist + timedelta(days=days_ahead))
    
    # Base
    df_dates = pd.DataFrame({'Date': dates_future})
    
    # Clima
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
    df_holidays = get_holidays_calendar(df_dates['Date'].min(), df_dates['Date'].max())
    df_dates = pd.merge(df_dates, df_holidays, on='Date', how='left')
    df_dates['IsHoliday'] = df_dates['IsHoliday'].fillna(0)

    # Merge Full
    unique_skus = df_hist[['SKU', 'Description']].drop_duplicates()
    unique_skus['key'] = 1; df_dates['key'] = 1
    df_master = pd.merge(df_dates, unique_skus, on='key').drop('key', axis=1)
    df_master = pd.merge(df_master, df_hist[['Date','SKU','Orders']], on=['Date','SKU'], how='left')
    
    # Preenche zeros, mas com cuidado:
    # Como cortamos 2022-2024 para Vero, esses dados vão aparecer como NaN no merge com df_dates antigos.
    # Vamos preencher com 0, o que é correto (para o modelo, é como se o produto não existisse antes).
    df_master.loc[df_master['Date'] <= end_date_hist, 'Orders'] = df_master.loc[df_master['Date'] <= end_date_hist, 'Orders'].fillna(0)

    # Features
    def gen_feat(d):
        d = d.sort_values(['SKU', 'Date'])
        d['DayOfWeek'] = d['Date'].dt.dayofweek
        d['IsWeekend'] = (d['DayOfWeek'] >= 5).astype(int)
        for l in [1,7,14]: d[f'lag_{l}'] = d.groupby('SKU')['Orders'].shift(l)
        d['rolling_mean_7'] = d.groupby('SKU')['Orders'].shift(1).rolling(7).mean()
        return d

    df_feat = gen_feat(df_master)
    # Importante: O dropna aqui vai remover 2022-2024 para Vero automaticamente, 
    # pois geramos features baseadas na data. O treino será focado em 2025+.
    train = df_feat[df_feat['Date'] <= end_date_hist].dropna()
    
    # Treino (XGBoost)
    feat_cols = ['DayOfWeek','IsWeekend','IsHoliday','Temp_Avg','Rain_mm','lag_1','lag_7','rolling_mean_7']
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, n_jobs=-1)
    model.fit(train[feat_cols], train['Orders'])

    # Loop Previsão
    preds = []
    curr = df_master[df_master['Date'] <= end_date_hist].copy()
    prog = st.progress(0)
    
    for i in range(1, days_ahead + 1):
        nxt = end_date_hist + timedelta(days=i)
        nxt_base = df_master[df_master['Date'] == nxt][['Date','SKU','Description','Temp_Avg','Rain_mm','IsHoliday']].copy()
        
        tmp = pd.concat([curr, nxt_base], sort=False)
        tmp = gen_feat(tmp)
        row = tmp[tmp['Date'] == nxt].copy()
        
        y = np.maximum(model.predict(row[feat_cols].fillna(0)), 0)
        row['Orders'] = y.round(0)
        
        preds.append(row)
        curr = pd.concat([curr, row], sort=False)
        prog.progress(i/days_ahead)
        
    return pd.concat(preds), df_hist_raw

# --- 5. INTERFACE ---
uploaded_file = st.file_uploader("📂 Carregue data.xlsx", type=['csv', 'xlsx'])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if not df_raw.empty:
        today = df_raw['Date'].max()
        st.info(f"Dados até: **{today.date()}** | Vero considerada a partir de: **01/01/2025**")
        
        if st.button("🚀 Gerar Previsão 14 Dias"):
            forecast, history = run_forecast(df_raw, days_ahead=14)
            
            # --- PIVOT PARA CSV FINAL (Formato Matriz) ---
            df_pivot = forecast.pivot_table(index=['SKU', 'Description'], 
                                          columns='Date', 
                                          values='Orders', 
                                          aggfunc='sum').reset_index()
            
            new_cols = []
            for c in df_pivot.columns:
                if isinstance(c, pd.Timestamp):
                    new_cols.append(c.strftime('%d/%m'))
                else:
                    new_cols.append(c)
            df_pivot.columns = new_cols

            # --- MÉTRICAS ---
            f_curr = forecast['Orders'].sum()
            
            # Comparativo
            fut_start = today + timedelta(days=1)
            fut_end = today + timedelta(days=14)
            
            ly_start = fut_start - timedelta(weeks=52)
            ly_end = fut_end - timedelta(weeks=52)
            f_ly = history[(history['Date'] >= ly_start) & (history['Date'] <= ly_end)]['Orders'].sum()

            l2y_start = fut_start - timedelta(weeks=104)
            l2y_end = fut_end - timedelta(weeks=104)
            f_2y = history[(history['Date'] >= l2y_start) & (history['Date'] <= l2y_end)]['Orders'].sum()

            st.divider()
            st.subheader("📊 Resumo de Performance")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Previsão Total (14 Dias)", f"{int(f_curr):,}")
            col2.metric("vs Ano Passado (2025)", f"{((f_curr/f_ly)-1)*100:.1f}%" if f_ly else "N/D")
            col3.metric("vs 2 Anos Atrás (2024)", f"{((f_curr/f_2y)-1)*100:.1f}%" if f_2y else "N/D")
            
            st.write("---")
            st.write("### 🗓️ Previsão Detalhada")
            st.dataframe(df_pivot)
            
            csv = df_pivot.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Planilha Detalhada", csv, "previsao_vero_2025plus.csv", "text/csv")