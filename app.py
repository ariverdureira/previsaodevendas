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

# --- 1. FUNÇÕES AUXILIARES (SIMPLIFICADAS) ---

def get_holidays_calendar(start_date, end_date):
    # Gera calendário de feriados SP
    try:
        br_holidays = holidays.Brazil(subdiv='SP', state='SP')
        date_range = pd.date_range(start_date, end_date)
        data = []
        for d in date_range:
            is_hol = 1 if d in br_holidays else 0
            data.append({'Date': d, 'IsHoliday': is_hol})
        return pd.DataFrame(data)
    except:
        # Fallback caso falhe a biblioteca
        d_range = pd.date_range(start_date, end_date)
        return pd.DataFrame({'Date': d_range, 'IsHoliday': 0})

def get_live_forecast(days=14, lat=-23.55, lon=-46.63):
    # Busca dados de clima
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
    if not isinstance(desc, str):
        return 'Outros'
    
    txt = desc.lower()
    
    # 1. Prioridade Absoluta
    if 'americana bola' in txt: 
        return 'Americana Bola'
    
    # 2. Linha Vero
    if any(x in txt for x in ['vero', 'primavera', 'roxa']): 
        return 'Vero'
    
    # 3. Minis
    if 'mini' in txt: 
        return 'Minis'
    
    # 4. Regra Especial: Insalata > 100g
    if 'insalata' in txt:
        # Procura por números antes de 'g'
        match = re.search(r'(\d+)\s*g', txt)
        if match:
            weight = int(match.group(1))
            if weight > 100:
                return 'Saladas'
    
    # 5. Legumes
    legumes = ['legume', 'cenoura', 'beterraba', 'abobrinha']
    if any(x in txt for x in legumes): 
        return 'Legumes'
    
    # 6. Saladas Gerais
    saladas = ['salada', 'alface', 'rúcula', 'agrião', 'insalata']
    if any(x in txt for x in saladas): 
        return 'Saladas'
    
    return 'Outros'

@st.cache_data
def load_data(uploaded_file):
    # Carregamento seguro
    try:
        # Tenta ler CSV ou Excel
        try: 
            df = pd.read_csv(uploaded_file, sep=',') 
        except: 
            df = pd.read_excel(uploaded_file)
        
        # Se separador for errado, tenta ponto e vírgula
        if df.shape[1] < 2: 
            df = pd.read_csv(uploaded_file, sep=';')
        
        df.columns = df.columns.str.strip()
        
        # Renomeia colunas padrão
        rename_map = {
            'Data':'Date', 'Dia':'Date',
            'Cod- SKU':'SKU', 'Código':'SKU',
            'Produto.DS_PRODUTO':'Description', 'Descrição':'Description',
            'Pedidos':'Orders', 'Qtde':'Orders'
        }
        df = df.rename(columns=rename_map)
        
        # Garante existência da descrição
        if 'Description' not in df.columns:
            if len(df.columns) >= 4:
                cols = ['Date','SKU','Description','Orders']
                # Pega as primeiras 4 colunas + o resto
                existing = list(df.columns)
                df.columns = cols + existing[4:]
            else:
                df['Description'] = 'Prod ' + df['SKU'].astype(str)

        # Converte tipos
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
        
        # Classifica
        df['Group'] = df['Description'].apply(classify_group)
        
        # Agrega
        return df.groupby(['Date','SKU','Description','Group'])['Orders'].sum().reset_index()
        
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return pd.DataFrame()

# --- 3. PRÉ-PROCESSAMENTO (LIMPEZA) ---

def filter_history_vero(df):
    # Filtra histórico antigo da Vero (apenas 2025+)
    mask_vero = df['Group'] == 'Vero'
    mask_date = df['Date'] >= '2025-01-01'
    
    # Mantém Vero recente OU Outros produtos (histórico todo)
    keep = (mask_vero & mask_date) | (~mask_vero)
    return df[keep].copy()

def clean_outliers(df):
    # Remove picos absurdos de implantação
    df = df.sort_values(['SKU', 'Date'])
    
    # Grupos alvo da limpeza
    targets = ['Vero', 'Americana Bola']
    skus_to_check = df[df['Group'].isin(targets)]['SKU'].unique()
    
    for sku in skus_to_check:
        mask = df['SKU'] == sku
        series = df.loc[mask, 'Orders']
        
        if len(series) < 5: 
            continue
            
        # Mediana móvel
        roll_med = series.rolling(window=14, min_periods=1, center=True).median()
        
        # Regra: Se valor > 4x a mediana local -> Substitui pela mediana
        is_outlier = series > (roll_med * 4)
        
        if is_outlier.any():
            # Substituição segura
            df.loc[mask & is_outlier, 'Orders'] = roll_med[is_outlier]
            
    return df

def generate_features(df):
    # Cria variáveis para o XGBoost
    d = df.sort_values(['SKU', 'Date']).copy()
    
    d['DayOfWeek'] = d['Date'].dt.dayofweek
    d['IsWeekend'] = (d['DayOfWeek'] >= 5).astype(int)
    
    # Lags (Vendas passadas)
    d['lag_1'] = d.groupby('SKU')['Orders'].shift(1)
    d['lag_7'] = d.groupby('SKU')['Orders'].shift(7)
    d['lag_14'] = d.groupby('SKU')['Orders'].shift(14)
    
    # Média Móvel
    d['roll_7'] = d.groupby('SKU')['Orders'].shift(1).rolling(7).mean()
    
    return d

# --- 4. MOTOR DE PREVISÃO ---

def run_forecast(df_raw, days_ahead=14):
    # 1. Aplica Filtros
    df_hist = filter_history_vero(df_raw)
    df_hist = clean_outliers(df_hist)
    
    # 2. Prepara Datas Futuras
    last_date = df_hist['Date'].max()
    start_date = df_hist['Date'].min()
    future_range = pd.date_range(start_date, last_date + timedelta(days=days_ahead))
    
    df_dates = pd.DataFrame({'Date': future_range})
    
    # 3. Clima e Feriados
    weather = get_live_forecast(days=days_ahead)
    
    # Clima Mock (caso API falhe ou para passado)
    np.random.seed(42)
    df_dates['Temp_Avg'] = np.random.normal(25, 3, len(df_dates))
    summer_months = [1, 2, 3, 12]
    is_summer = df_dates['Date'].dt.month.isin(summer_months)
    df_dates['Rain_mm'] = np.where(is_summer, np.random.exponential(8, len(df_dates)), 4)
    
    # Merge Clima Real se existir
    if weather is not None:
        weather['Date'] = pd.to_datetime(weather['Date'])
        df_dates = pd.merge(df_dates, weather, on='Date', how='left', suffixes=('', '_real'))
        df_dates['Temp_Avg'] = df_dates['Temp_Avg_real'].fillna(df_dates['Temp_Avg'])
        df_dates['Rain_mm'] = df_dates['Rain_mm_real'].fillna(df_dates['Rain_mm'])
        df_dates = df_dates[['Date', 'Temp_Avg', 'Rain_mm']]
        
    # Merge Feriados
    holidays_df = get_holidays_calendar(df_dates['Date'].min(), df_dates['Date'].max())
    df_dates = pd.merge(df_dates, holidays_df, on='Date', how='left')
    df_dates['IsHoliday'] = df_dates['IsHoliday'].fillna(0)
    
    # 4. Cross Join (SKU x Data)
    unique_skus = df_hist[['SKU', 'Description', 'Group']].drop_duplicates()
    unique_skus['key'] = 1
    df_dates['key'] = 1
    
    df_master = pd.merge(df_dates, unique_skus, on='key').drop('key', axis=1)
    
    # 5. Junta Vendas Reais
    df_master = pd.merge(df_master, df_hist[['Date','SKU','Orders']], on=['Date','SKU'], how='left')
    
    # Preenche zeros apenas no passado
    mask_past = df_master['Date'] <= last_date
    df_master.loc[mask_past, 'Orders'] = df_master.loc[mask_past, 'Orders'].fillna(0)
    
    # 6. Feature Engineering
    df_feat = generate_features(df_master)
    
    # Treino (apenas passado)
    train_data = df_feat[df_feat['Date'] <= last_date].dropna()
    
    if train_data.empty:
        st.error("Erro: Dados insuficientes para treinamento.")
        return pd.DataFrame(), df_raw
        
    # 7. Treinamento Modelo
    features = ['DayOfWeek','IsWeekend','IsHoliday','Temp_Avg','Rain_mm','lag_1','lag_7','roll_7']
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, n_jobs=-1)
    model.fit(train_data[features], train_data['Orders'])
    
    # 8. Loop de Previsão
    preds = []
    current_df = df_master[df_master['Date'] <= last_date].copy()
    
    progress_bar = st.progress(0)
    
    for i in range(1, days_ahead + 1):
        next_day = last_date + timedelta(days=i)
        
        # Prepara a base do dia seguinte
        cols_static = ['Date','SKU','Description','Group','Temp_Avg','Rain_mm','IsHoliday']
        next_base = df_master[df_master['Date'] == next_day][cols_static].copy()
        
        # Recalcula features com o dia novo
        temp_full = pd.concat([current_df, next_base], sort=False)
        temp_full = generate_features(temp_full)
        
        # Seleciona linha para prever
        row_pred = temp_full[temp_full['Date'] == next_day].copy()
        
        # Predição
        X_test = row_pred[features].fillna(0)
        y_pred = model.predict(X_test)
        
        # Trata resultado
        row_pred['Orders'] = np.maximum(np.round(y_pred, 0), 0)
        
        # --- REGRAS DE NEGÓCIO FINAIS ---
        # 1. Domingo = 0
        is_sunday = next_day.dayofweek == 6
        # 2. Feriado = 0
        is_holiday = row_pred['IsHoliday'].values[0] == 1
        
        if is_sunday or is_holiday:
            row_pred['Orders'] = 0
            
        preds.append(row_pred)
        current_df = pd.concat([current_df, row_pred], sort=False)
        
        progress_bar.progress(i / days_ahead)
        
    return pd.concat(preds), df_hist

# --- 5. INTERFACE DO USUÁRIO ---

uploaded_file = st.file_uploader("📂 Carregue seu arquivo Excel/CSV", type=['csv', 'xlsx'])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if not df_raw.empty:
        max_date = df_raw['Date'].max()
        st.info(f"Dados carregados até: **{max_date.date()}**")
        
        if st.button("🚀 Gerar Previsão"):
            with st.spinner("Processando Inteligência Artificial..."):
                try:
                    forecast, history = run_forecast(df_raw, days_ahead=14)
                    
                    if not forecast.empty:
                        # --- CÁLCULOS COMPARATIVOS ---
                        f_start = max_date + timedelta(days=1)
                        f_end = max_date + timedelta(days=14)
                        
                        # Janelas de Tempo Anteriores
                        ly_start = f_start - timedelta(weeks=52)
                        ly_end = f_end - timedelta(weeks=52)
                        l2y_start = f_start - timedelta(weeks=104)
                        l2y_end = f_end - timedelta(weeks=104)
                        
                        # Filtra bases de comparação
                        hist_ly = history[(history['Date'] >= ly_start) & (history['Date'] <= ly_end)]
                        hist_2y = history[(history['Date'] >= l2y_start) & (history['Date'] <= l2y_end)]
                        
                        groups = ['Americana Bola', 'Vero', 'Saladas', 'Legumes', 'Minis', 'Outros']
                        summary = []
                        
                        # Loop Grupos
                        for g in groups:
                            v_curr = forecast[forecast['Group'] == g]['Orders'].sum()
                            v_ly = hist_ly[hist_ly['Group'] == g]['Orders'].sum()
                            v_2y = hist_2y[hist_2y['Group'] == g]['Orders'].sum()
                            
                            # Variação %
                            p_ly = ((v_curr / v_ly) - 1) * 100 if v_ly > 0 else 0
                            p_2y = ((v_curr / v_2y) - 1) * 100 if v_2y > 0 else 0
                            
                            summary.append({
                                'Grupo': g,
                                'Previsão 14d': int(v_curr),
                                '2025': int(v_ly),
                                'Var % (25)': f"{p_ly:+.1f}%",
                                '2024': int(v_2y),
                                'Var % (24)': f"{p_2y:+.1f}%"
                            })
                            
                        # --- LINHA DE TOTAL ---
                        tot_cur = forecast['Orders'].sum()
                        tot_ly = hist_ly['Orders'].sum()
                        tot_2y = hist_2y['Orders'].sum()
                        
                        pt_ly = ((tot_cur / tot_ly) - 1) * 100 if tot_ly > 0 else 0
                        pt_2y = ((tot_cur / tot_2y) - 1) * 100 if tot_2y > 0 else 0
                        
                        summary.append({
                            'Grupo': 'TOTAL GERAL',
                            'Previsão 14d': int(tot_cur),
                            '2025': int(tot_ly),
                            'Var % (25)': f"{pt_ly:+.1f}%",
                            '