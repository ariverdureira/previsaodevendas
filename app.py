import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import plotly.express as px
import requests
import holidays

st.set_page_config(page_title="Forecasting Completo", layout="wide")
st.title("📊 Previsão de Vendas (14 Dias) & Comparativo 2 Anos")

# --- 1. FUNÇÕES AUXILIARES ---
def get_holidays_calendar(start_date, end_date):
    br_holidays = holidays.Brazil(subdiv='SP', state='SP')
    date_range = pd.date_range(start_date, end_date)
    return pd.DataFrame([{'Date': d, 'IsHoliday': 1 if d in br_holidays else 0, 
                          'HolidayName': br_holidays.get(d)} for d in date_range])

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

# --- 3. MOTOR DE PREVISÃO ---
def run_forecast(df_hist, days_ahead=14):
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
    train = df_feat[df_feat['Date'] <= end_date_hist].dropna()
    
    # Treino
    feat_cols = ['DayOfWeek','IsWeekend','IsHoliday','Temp_Avg','Rain_mm','lag_1','lag_7','rolling_mean_7']
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1)
    model.fit(train[feat_cols], train['Orders'])

    # Lift Vero (Com trava de segurança para picos)
    last_day = end_date_hist
    compare_day = end_date_hist - timedelta(days=7)
    
    vero_mask = df_hist['Description'].str.lower().str.contains('vero|primavera|roxa', regex=True)
    vero_skus = df_hist[vero_mask]['SKU'].unique()
    
    lift_factors = {}
    for sku in vero_skus:
        sales_new = df_hist[(df_hist['Date'] == last_day) & (df_hist['SKU'] == sku)]['Orders'].sum()
        sales_old = df_hist[(df_hist['Date'] == compare_day) & (df_hist['SKU'] == sku)]['Orders'].sum()
        
        if sales_new > 0 and sales_old > 10:
            factor = sales_new / sales_old
            factor = min(factor, 2.5) # Limita picos artificiais
            factor = max(factor, 1.0)
        else:
            factor = 1.0
        lift_factors[sku] = factor

    # Loop Previsão
    preds = []
    curr = df_master[df_master['Date'] <= end_date_hist].copy()
    prog = st.progress(0)
    
    for i in range(1, days_ahead + 1):
        nxt = end_date_hist + timedelta(days=i)
        nxt_base = df_master[df_master['Date'] == nxt][['Date','SKU','Description','Temp_Avg','Rain_mm','IsHoliday','HolidayName']].copy()
        
        tmp = pd.concat([curr, nxt_base], sort=False)
        tmp = gen_feat(tmp)
        row = tmp[tmp['Date'] == nxt].copy()
        
        y = np.maximum(model.predict(row[feat_cols].fillna(0)), 0)
        row['Orders'] = y
        
        for sku, fac in lift_factors.items():
            row.loc[row['SKU']==sku, 'Orders'] *= fac
            
        row['Orders'] = row['Orders'].round(0)
        preds.append(row)
        curr = pd.concat([curr, row], sort=False)
        prog.progress(i/days_ahead)
        
    return pd.concat(preds), df_hist

# --- 4. INTERFACE ---
uploaded_file = st.file_uploader("📂 Carregue data.xlsx", type=['csv', 'xlsx'])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if not df_raw.empty:
        today = df_raw['Date'].max()
        st.info(f"Dados até: **{today.date()}**")
        
        if st.button("🚀 Gerar Previsão 14 Dias"):
            forecast, history = run_forecast(df_raw, days_ahead=14)
            
            # --- CÁLCULO DAS MÉTRICAS COMPARATIVAS ---
            
            # 1. Passado (Últimas 3 semanas)
            past_start = today - timedelta(days=20)
            p_curr = history[(history['Date']>=past_start) & (history['Date']<=today)]['Orders'].sum()
            p_ly = history[(history['Date']>=past_start-timedelta(weeks=52)) & (history['Date']<=today-timedelta(weeks=52))]['Orders'].sum()
            p_2y = history[(history['Date']>=past_start-timedelta(weeks=104)) & (history['Date']<=today-timedelta(weeks=104))]['Orders'].sum()
            
            # 2. Futuro (Próximas 2 semanas)
            fut_end = today + timedelta(days=14)
            f_curr = forecast['Orders'].sum()
            
            # Comparativo Futuro vs Ano Passado (LY - 52 semanas)
            f_ly = history[(history['Date']>today-timedelta(weeks=52)) & (history['Date']<=today-timedelta(weeks=52)+timedelta(days=14))]['Orders'].sum()
            
            # Comparativo Futuro vs 2 Anos Atrás (2Y - 104 semanas) -> A PARTE QUE FALTOU ANTES
            f_2y = history[(history['Date']>today-timedelta(weeks=104)) & (history['Date']<=today-timedelta(weeks=104)+timedelta(days=14))]['Orders'].sum()

            # --- EXIBIÇÃO ---
            st.divider()
            st.subheader("📊 Relatório de Performance e Tendência")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔙 Realizado (Últimos 21 Dias)")
                st.metric("Volume Total", f"{int(p_curr):,}")
                st.metric("Cresc. vs Ano Passado", f"{((p_curr/p_ly)-1)*100:.1f}%")
                st.metric("Cresc. vs 2 Anos Atrás", f"{((p_curr/p_2y)-1)*100:.1f}%")
            
            with col2:
                st.markdown("#### 🔜 Previsão (Próximos 14 Dias)")
                st.metric("Volume Projetado", f"{int(f_curr):,}")
                
                # Exibe variação vs Ano Passado (se houver dados)
                if f_ly > 0:
                    st.metric("Cresc. Projetado vs Ano Passado", f"{((f_curr/f_ly)-1)*100:.1f}%")
                else:
                    st.metric("Cresc. vs Ano Passado", "N/D")
                    
                # Exibe variação vs 2 Anos Atrás (NOVO)
                if f_2y > 0:
                    st.metric("Cresc. Projetado vs 2 Anos Atrás", f"{((f_curr/f_2y)-1)*100:.1f}%")
                else:
                    st.metric("Cresc. vs 2 Anos Atrás", "N/D")
            
            st.divider()
            
            # Gráfico e Download
            sel_sku = st.selectbox("Produto:", forecast['Description'].unique())
            fig = px.line(forecast[forecast['Description']==sel_sku], x='Date', y='Orders', markers=True, title=sel_sku)
            st.plotly_chart(fig, use_container_width=True)
            
            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar CSV (14 Dias)", csv, "previsao_vendas_ajustada.csv", "text/csv")