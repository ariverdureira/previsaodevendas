import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays
import traceback
import re
from google import genai
from sklearn.metrics import mean_absolute_error

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="PCP Verdureira", layout="wide")

# ==============================================================================
# 1. FUNÇÕES AUXILIARES E CALENDÁRIO INTELIGENTE
# ==============================================================================

def get_holidays_calendar(start_date, end_date):
    """
    Gera calendário contendo:
    1. Feriados Nacionais (SP)
    2. Datas Comemorativas Móveis (Dia das Mães, Pais)
    3. Flag de 'Alto Impacto' para datas de muito consumo
    """
    try:
        # 1. Feriados Oficiais
        br_holidays = holidays.Brazil(subdiv='SP', state='SP')
        
        # Datas de Alto Impacto (Fixo)
        high_impact_fixed = {
            (12, 25): "Natal",
            (1, 1): "Ano Novo"
        }
        
        data = []
        # Garante iteração por todos os anos do range
        years = list(range(start_date.year, end_date.year + 1))
        
        # 2. Calcula Datas Móveis (Mães e Pais)
        mothers_days = []
        fathers_days = []
        
        for year in years:
            # Dia das Mães: 2º Domingo de Maio
            may_sundays = pd.date_range(start=f'{year}-05-01', end=f'{year}-05-31', freq='W-SUN')
            if len(may_sundays) >= 2:
                mothers_days.append(may_sundays[1].date())
            
            # Dia dos Pais: 2º Domingo de Agosto
            aug_sundays = pd.date_range(start=f'{year}-08-01', end=f'{year}-08-31', freq='W-SUN')
            if len(aug_sundays) >= 2:
                fathers_days.append(aug_sundays[1].date())

        # Gera o DataFrame dia a dia
        date_range = pd.date_range(start_date, end_date)
        
        for d in date_range:
            d_date = d.date()
            is_hol = 0
            is_high = 0
            
            # Verifica Feriado Oficial
            if d_date in br_holidays:
                is_hol = 1
                # Se for Natal ou Ano Novo, é Alto Impacto
                if (d.month, d.day) in high_impact_fixed:
                    is_high = 1
            
            # Verifica Mães e Pais (Alto Impacto)
            if d_date in mothers_days or d_date in fathers_days:
                is_hol = 1 # Consideramos como feriado/evento
                is_high = 1 # E de alto impacto
            
            # Verifica Páscoa (Aproximação via feriados cristãos se disponível ou lógica manual)
            # A lib holidays geralmente traz 'Sexta-feira Santa' e 'Corpus Christi'
            # Vamos considerar Sexta-Santa como gatilho de alto impacto também
            if d_date in br_holidays and br_holidays.get(d_date) == "Sexta-feira Santa":
                 is_high = 1

            data.append({
                'Date': d, 
                'IsHoliday': is_hol,
                'IsHighImpact': is_high
            })
            
        return pd.DataFrame(data)
    except:
        # Fallback seguro
        d_range = pd.date_range(start_date, end_date)
        return pd.DataFrame({'Date': d_range, 'IsHoliday': 0, 'IsHighImpact': 0})

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
        r = requests.get(url, params=params, timeout=5).json()
        
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

def classify_group(desc):
    if not isinstance(desc, str): return 'Outros'
    txt = desc.lower()
    
    if 'americana bola' in txt: return 'Americana Bola'
    
    vero_keys = ['vero', 'primavera', 'roxa', 'mix', 'repolho', 'couve', 'rucula hg']
    if any(x in txt for x in vero_keys): return 'Vero'
    
    if 'mini' in txt: return 'Minis'
    
    if 'insalata' in txt:
        match = re.search(r'(\d+)\s*g', txt)
        if match:
            weight = int(match.group(1))
            if weight > 100: return 'Saladas'
    
    legumes_keys = [
        'legume', 'cenoura', 'beterraba', 'abobrinha',
        'batata', 'mandioca', 'mandioquinha', 'sopa',
        'grao de bico', 'grão de bico', 'grao-de-bico', 'grão-de-bico',
        'lentilha', 'pinhao', 'pinhão', 'quinoa', 'milho'
    ]
    if any(x in txt for x in legumes_keys): return 'Legumes'
    
    saladas_keys = ['salada', 'alface', 'rúcula', 'rucula', 'agrião', 'agriao', 'insalata']
    if any(x in txt for x in saladas_keys): return 'Saladas'
    
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

# ==============================================================================
# 2. ENGENHARIA DE FEATURES
# ==============================================================================

def filter_history_vero(df):
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
    
    # Efeito Pagamento (Sazonalidade Mensal)
    d['DayOfMonth'] = d['Date'].dt.day
    d['IsPaydayWeek'] = d['DayOfMonth'].between(5, 10).astype(int)
    
    # Lags
    d['lag_1'] = d.groupby('SKU')['Orders'].shift(1)
    d['lag_7'] = d.groupby('SKU')['Orders'].shift(7)
    d['lag_14'] = d.groupby('SKU')['Orders'].shift(14)
    d['roll_7'] = d.groupby('SKU')['Orders'].shift(1).rolling(7).mean()
    
    return d

# ==============================================================================
# 3. MOTOR DE PREVISÃO
# ==============================================================================

def run_forecast(df_raw, days_ahead=14):
    df_train_base = filter_history_vero(df_raw)
    df_train_base = clean_outliers(df_train_base)
    
    last_date = df_train_base['Date'].max()
    start_date = df_train_base['Date'].min()
    future_range = pd.date_range(start_date, last_date + timedelta(days=days_ahead))
    df_dates = pd.DataFrame({'Date': future_range})
    
    # --- CLIMA ---
    weather = get_live_forecast(days=days_ahead)
    
    # Fallback Sintético
    np.random.seed(42)
    df_dates['Temp_Avg'] = np.random.normal(25, 3, len(df_dates))
    is_summer = df_dates['Date'].dt.month.isin([1, 2, 3, 12])
    df_dates['Rain_mm'] = np.where(is_summer, np.random.exponential(8, len(df_dates)), 4)
    
    if weather is not None:
        weather['Date'] = pd.to_datetime(weather['Date'])
        df_dates = pd.merge(df_dates, weather, on='Date', how='left', suffixes=('', '_real'))
        df_dates['Temp_Avg'] = df_dates['Temp_Avg_real'].fillna(df_dates['Temp_Avg'])
        df_dates['Rain_mm'] = df_dates['Rain_mm_real'].fillna(df_dates['Rain_mm'])
        df_dates = df_dates[['Date', 'Temp_Avg', 'Rain_mm']]
    
    weather_future = df_dates[
        (df_dates['Date'] > last_date) & 
        (df_dates['Date'] <= last_date + timedelta(days=days_ahead))
    ][['Date', 'Temp_Avg', 'Rain_mm']].copy()
    
    # --- FERIADOS E EVENTOS ESPECIAIS ---
    holidays_df = get_holidays_calendar(df_dates['Date'].min(), df_dates['Date'].max())
    df_dates = pd.merge(df_dates, holidays_df, on='Date', how='left')
    df_dates['IsHoliday'] = df_dates['IsHoliday'].fillna(0)
    df_dates['IsHighImpact'] = df_dates['IsHighImpact'].fillna(0)
    
    # --- VÉSPERA DE IMPACTO (O "Pulo do Gato") ---
    # Se amanhã é dia de Alto Impacto (Mães, Natal), hoje ganha flag 1
    df_dates['IsHighImpactNextDay'] = df_dates['IsHighImpact'].shift(-1).fillna(0)
    
    # --- MERGE ---
    unique_skus = df_train_base[['SKU', 'Description', 'Group']].drop_duplicates()
    unique_skus['key'] = 1
    df_dates['key'] = 1
    
    df_master = pd.merge(df_dates, unique_skus, on='key').drop('key', axis=1)
    df_master = pd.merge(df_master, df_train_base[['Date','SKU','Orders']], on=['Date','SKU'], how='left')
    
    mask_past = df_master['Date'] <= last_date
    df_master.loc[mask_past, 'Orders'] = df_master.loc[mask_past, 'Orders'].fillna(0)
    
    df_feat = generate_features(df_master)
    
    # Features atualizadas para incluir a Véspera de Alto Impacto
    features = [
        'DayOfWeek', 'IsWeekend', 
        'IsHoliday', 'IsHighImpact', 'IsHighImpactNextDay', # <--- Novas Flags
        'DayOfMonth', 'IsPaydayWeek', 
        'Temp_Avg', 'Rain_mm', 
        'lag_1', 'lag_7', 'roll_7'
    ]
    
    train_full = df_feat[df_feat['Date'] <= last_date].dropna()
    
    if train_full.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --- MINI AUTO-ML ---
    split_date = last_date - timedelta(days=14)
    X_train_sub = train_full[train_full['Date'] <= split_date]
    X_val_sub = train_full[train_full['Date'] > split_date]
    
    best_params = {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 6}
    
    if not X_train_sub.empty and not X_val_sub.empty:
        configs = [
            {'name': 'Rápido', 'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5},
            {'name': 'Profundo', 'n_estimators': 400, 'learning_rate': 0.02, 'max_depth': 7},
            {'name': 'Padrão', 'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 6}
        ]
        
        best_mae = float('inf')
        
        for cfg in configs:
            try:
                m = XGBRegressor(
                    n_estimators=cfg['n_estimators'], 
                    learning_rate=cfg['learning_rate'], 
                    max_depth=cfg['max_depth'], 
                    n_jobs=-1
                )
                m.fit(X_train_sub[features], X_train_sub['Orders'])
                val_preds = m.predict(X_val_sub[features])
                mae = mean_absolute_error(X_val_sub['Orders'], val_preds)
                
                if mae < best_mae:
                    best_mae = mae
                    best_params = cfg
            except: continue
        print(f"Config Vencedora: {best_params.get('name')}")

    model = XGBRegressor(
        n_estimators=best_params['n_estimators'], 
        learning_rate=best_params['learning_rate'], 
        max_depth=best_params['max_depth'], 
        n_jobs=-1
    )
    model.fit(train_full[features], train_full['Orders'])
    
    # --- LOOP DE PREVISÃO ---
    preds = []
    current_df = df_master[df_master['Date'] <= last_date].copy()
    progress_bar = st.progress(0)
    
    for i in range(1, days_ahead + 1):
        next_day = last_date + timedelta(days=i)
        
        cols_static = ['Date','SKU','Description','Group','Temp_Avg','Rain_mm',
                       'IsHoliday','IsHighImpact','IsHighImpactNextDay']
        next_base = df_master[df_master['Date'] == next_day][cols_static].copy()
        
        temp_full = pd.concat([current_df, next_base], sort=False)
        temp_full = generate_features(temp_full)
        
        row_pred = temp_full[temp_full['Date'] == next_day].copy()
        
        X_test = row_pred[features].fillna(0)
        y_pred = model.predict(X_test)
        
        row_pred['Orders'] = np.maximum(np.round(y_pred, 0), 0)
        
        is_sunday = next_day.dayofweek == 6
        is_holiday_today = row_pred['IsHoliday'].values[0] == 1
        
        if is_sunday or is_holiday_today:
            row_pred['Orders'] = 0
            
        preds.append(row_pred)
        current_df = pd.concat([current_df, row_pred], sort=False)
        progress_bar.progress(i / days_ahead)
        
    return pd.concat(preds), weather_future

# ==============================================================================
# 4. INTERFACE VISUAL
# ==============================================================================

st.markdown("""
    <style>
        .title-text {
            text-align: center;
            color: #8CFF00;
            font-family: sans-serif;
            font-weight: bold;
            font-size: 3rem;
            margin-bottom: 2rem;
        }
        .stAppHeader {
            background-color: transparent;
        }
    </style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    try:
        st.image("AF-VERDUREIRA-LOGO-HORIZONTAL-07.jpg", use_container_width=True)
    except:
        st.warning("⚠️ Logo (JPG) não encontrada.")

st.markdown('<h1 class="title-text">PCP - Previsão de Vendas</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Carregue seu arquivo Excel/CSV", type=['csv', 'xlsx'])

if 'last_file' not in st.session_state: st.session_state.last_file = None
if uploaded_file and uploaded_file != st.session_state.last_file:
    st.session_state.clear()
    st.session_state.last_file = uploaded_file

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if not df_raw.empty:
        max_date = df_raw['Date'].max()
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            processar = st.button("🚀 Gerar Previsão", use_container_width=True)

        if processar:
            with st.spinner("Analisando Feriados, Clima e Sazonalidade..."):
                try:
                    # HORIZONTE 7 DIAS (Mais Seguro para Perecíveis)
                    forecast_result, weather_result = run_forecast(df_raw, days_ahead=7)
                    
                    if not forecast_result.empty:
                        st.session_state['forecast_data'] = forecast_result
                        st.session_state['weather_data'] = weather_result
                        st.session_state['has_run'] = True
                except Exception as e:
                    st.error(f"Erro no cálculo: {e}")
                    st.write(traceback.format_exc())

        if st.session_state.get('has_run', False):
            forecast = st.session_state['forecast_data']
            weather_df = st.session_state.get('weather_data', pd.DataFrame())
            
            # --- 1. PREVISÃO DO TEMPO ---
            if not weather_df.empty:
                st.divider()
                st.subheader("🌤️ Previsão do Tempo (Próximos 7 Dias)")
                st.caption("Dados meteorológicos utilizados pelo modelo.")
                
                w_disp = weather_df.head(7).copy()
                w_disp['Date'] = w_disp['Date'].dt.strftime('%d/%m')
                w_disp = w_disp.rename(columns={
                    'Date': 'Data', 
                    'Temp_Avg': 'Temp. Média (°C)', 
                    'Rain_mm': 'Chuva (mm)'
                })
                w_disp['Temp. Média (°C)'] = w_disp['Temp. Média (°C)'].map('{:.1f}'.format)
                w_disp['Chuva (mm)'] = w_disp['Chuva (mm)'].map('{:.1f}'.format)
                
                st.dataframe(w_disp, hide_index=True, use_container_width=True)
            else:
                st.warning("⚠️ Dados climáticos estimados (API indisponível).")
            
            # --- 2. RESUMO EXECUTIVO ---
            f_start = max_date + timedelta(days=1)
            ly_start = f_start - timedelta(weeks=52)
            ly_end = ly_start + timedelta(days=6) # Ajustado para 7d
            l2y_start = f_start - timedelta(weeks=104)
            l2y_end = l2y_start + timedelta(days=6) # Ajustado para 7d
            
            hist_ly = df_raw[(df_raw['Date'] >= ly_start) & (df_raw['Date'] <= ly_end)]
            hist_2y = df_raw[(df_raw['Date'] >= l2y_start) & (df_raw['Date'] <= l2y_end)]
            
            groups = ['Americana Bola', 'Vero', 'Saladas', 'Legumes', 'Minis']
            summary = []
            
            for g in groups:
                v_curr = forecast[forecast['Group'] == g]['Orders'].sum()
                v_ly = hist_ly[hist_ly['Group'] == g]['Orders'].sum()
                v_2y = hist_2y[hist_2y['Group'] == g]['Orders'].sum()
                
                p_ly = ((v_curr / v_ly) - 1) * 100 if v_ly > 0 else 0
                p_2y = ((v_curr / v_2y) - 1) * 100 if v_2y > 0 else 0
                
                summary.append({
                    'Grupo': g,
                    'Previsão 7d': int(v_curr),
                    '2025 (Mesmo Período)': int(v_ly),
                    'Var % (vs 25)': f"{p_ly:+.1f}%",
                    '2024 (Mesmo Período)': int(v_2y),
                    'Var % (vs 24)': f"{p_2y:+.1f}%"
                })
            
            tot_cur = forecast['Orders'].sum()
            tot_ly = hist_ly['Orders'].sum()
            tot_2y = hist_2y['Orders'].sum()
            
            pt_ly = ((tot_cur / tot_ly) - 1) * 100 if tot_ly > 0 else 0
            pt_2y = ((tot_cur / tot_2y) - 1) * 100 if tot_2y > 0 else 0
            
            summary.append({
                'Grupo': 'TOTAL GERAL',
                'Previsão 7d': int(tot_cur),
                '2025 (Mesmo Período)': int(tot_ly),
                'Var % (vs 25)': f"{pt_ly:+.1f}%",
                '2024 (Mesmo Período)': int(tot_2y),
                'Var % (vs 24)': f"{pt_2y:+.1f}%"
            })
            
            st.divider()
            st.subheader("📊 Resumo Executivo")
            df_summary = pd.DataFrame(summary)
            st.dataframe(df_summary, hide_index=True, use_container_width=True)
            
            # --- 3. TABELA DETALHADA ---
            df_piv = forecast.pivot_table(
                index=['SKU', 'Description', 'Group'], 
                columns='Date', 
                values='Orders', 
                aggfunc='sum'
            ).reset_index()
            
            num_cols = df_piv.select_dtypes(include=[np.number]).columns
            total_data = df_piv[num_cols].sum().to_dict()
            total_data['SKU'] = 'TOTAL'
            total_data['Description'] = 'TOTAL GERAL'
            total_data['Group'] = '-'
            
            df_total = pd.DataFrame([total_data])
            df_piv = pd.concat([df_piv, df_total], ignore_index=True)
            
            cols_fmt = []
            for c in df_piv.columns:
                if isinstance(c, pd.Timestamp):
                    cols_fmt.append(c.strftime('%d/%m'))
                else:
                    cols_fmt.append(c)
            df_piv.columns = cols_fmt
            
            st.write("### 🗓️ Previsão Detalhada")
            st.dataframe(df_piv, use_container_width=True)
            
            # --- DOWNLOAD ---
            d_inicial = forecast['Date'].min().strftime('%d-%m-%Y')
            d_final = forecast['Date'].max().strftime('%d-%m-%Y')
            nome_arquivo = f"Previsao_{d_inicial}_a_{d_final}.csv"
            
            csv = df_piv.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Planilha", csv, nome_arquivo, "text/csv")

            # --- 4. IA GEMINI ---
            st.divider()
            st.subheader("🤖 Analista IA")
            
            api_key = st.text_input("Insira sua Gemini API Key:", type="password")
            
            if api_key:
                try:
                    client = genai.Client(api_key=api_key)
                    
                    tabela_anual_str = df_summary.to_string(index=False)

                    dt_cut = max_date - timedelta(days=60)
                    df_h_recent = df_raw[df_raw['Date'] > dt_cut]
                    days_hist = (df_h_recent['Date'].max() - df_h_recent['Date'].min()).days + 1
                    days_hist = max(days_hist, 1)
                    
                    media_hist = df_h_recent.groupby('Group')['Orders'].sum() / days_hist
                    days_fore = 7
                    media_fore = forecast.groupby('Group')['Orders'].sum() / days_fore
                    
                    df_ritmo = pd.DataFrame({
                        'Média Diária (Últimos 60d)': media_hist,
                        'Média Diária (Prevista 7d)': media_fore
                    })
                    
                    df_ritmo['Aceleração de Vendas (%)'] = ((df_ritmo['Média Diária (Prevista 7d)'] / df_ritmo['Média Diária (Últimos 60d)']) - 1) * 100
                    tabela_ritmo_str = df_ritmo.round(1).to_string()
                    
                    top_sku = forecast.groupby('Description')['Orders'].sum().nlargest(5).to_string()
                    
                    st.info("Conectado.")
                    query = st.text_area("Pergunta:", key="gemini_query")
                    
                    if st.button("Consultar IA"):
                        with st.spinner("Analisando..."):
                            prompt = f"""
                            Você é um analista sênior.
                            TABELA 1: COMPARATIVO ANUAL (Volume Total)
                            {tabela_anual_str}
                            TABELA 2: RITMO DE VENDAS (Média Diária)
                            {tabela_ritmo_str}
                            TABELA 3: TOP PRODUTOS
                            {top_sku}
                            Pergunta: {query}
                            Responda em português.
                            """
                            response = client.models.generate_content(
                                model="gemini-1.5-flash",
                                contents=prompt
                            )
                            st.markdown(response.text)
                            
                except Exception as e:
                    st.error(f"Erro Conexão IA: {e}")