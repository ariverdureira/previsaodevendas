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
    """Gera calendário de feriados e datas comemorativas"""
    try:
        br_holidays = holidays.Brazil(subdiv='SP', state='SP')
        high_impact_fixed = {(12, 25): "Natal", (1, 1): "Ano Novo"}
        
        data = []
        years = list(range(start_date.year, end_date.year + 1))
        mothers_days = []
        fathers_days = []
        
        for year in years:
            may_sundays = pd.date_range(start=f'{year}-05-01', end=f'{year}-05-31', freq='W-SUN')
            if len(may_sundays) >= 2: mothers_days.append(may_sundays[1].date())
            aug_sundays = pd.date_range(start=f'{year}-08-01', end=f'{year}-08-31', freq='W-SUN')
            if len(aug_sundays) >= 2: fathers_days.append(aug_sundays[1].date())

        date_range = pd.date_range(start_date, end_date)
        
        for d in date_range:
            d_date = d.date()
            is_hol = 0
            is_high = 0
            
            if d_date in br_holidays:
                is_hol = 1
                if (d.month, d.day) in high_impact_fixed: is_high = 1
            
            if d_date in mothers_days or d_date in fathers_days:
                is_hol = 1 
                is_high = 1 
            
            if d_date in br_holidays and br_holidays.get(d_date) == "Sexta-feira Santa":
                 is_high = 1

            data.append({'Date': d, 'IsHoliday': is_hol, 'IsHighImpact': is_high})
            
        return pd.DataFrame(data)
    except:
        d_range = pd.date_range(start_date, end_date)
        return pd.DataFrame({'Date': d_range, 'IsHoliday': 0, 'IsHighImpact': 0})

@st.cache_data(ttl=3600)
def get_weather_data(start_date, end_date, lat=-23.55, lon=-46.63):
    """Busca Clima Real (Passado) e Previsão (Futuro)"""
    try:
        today = pd.Timestamp.now().normalize()
        hist_end = min(end_date, today - timedelta(days=2))
        df_hist = pd.DataFrame()
        
        if start_date < hist_end:
            url_hist = "https://archive-api.open-meteo.com/v1/archive"
            params_hist = {
                "latitude": lat, "longitude": lon,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": hist_end.strftime('%Y-%m-%d'),
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "timezone": "America/Sao_Paulo"
            }
            r_hist = requests.get(url_hist, params=params_hist, timeout=5).json()
            if 'daily' in r_hist:
                dates = pd.to_datetime(r_hist['daily']['time'])
                t_avg = (np.array(r_hist['daily']['temperature_2m_max']) + np.array(r_hist['daily']['temperature_2m_min'])) / 2
                df_hist = pd.DataFrame({'Date': dates, 'Temp_Avg': t_avg, 'Rain_mm': r_hist['daily']['precipitation_sum']})

        df_fore = pd.DataFrame()
        if end_date >= today:
            url_fore = "https://api.open-meteo.com/v1/forecast"
            params_fore = {
                "latitude": lat, "longitude": lon,
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "timezone": "America/Sao_Paulo",
                "forecast_days": 16 
            }
            r_fore = requests.get(url_fore, params=params_fore, timeout=5).json()
            if 'daily' in r_fore:
                dates = pd.to_datetime(r_fore['daily']['time'])
                t_avg = (np.array(r_fore['daily']['temperature_2m_max']) + np.array(r_fore['daily']['temperature_2m_min'])) / 2
                df_fore = pd.DataFrame({'Date': dates, 'Temp_Avg': t_avg, 'Rain_mm': r_fore['daily']['precipitation_sum']})

        df_full = pd.concat([df_hist, df_fore]).drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
        df_full = df_full[(df_full['Date'] >= start_date) & (df_full['Date'] <= end_date)]
        return df_full
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
    legumes_keys = ['legume', 'cenoura', 'beterraba', 'abobrinha', 'batata', 'mandioca', 'mandioquinha', 'sopa', 'grao de bico', 'grão de bico', 'lentilha', 'pinhao', 'pinhão', 'quinoa', 'milho']
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
        rename_map = {'Data':'Date', 'Dia':'Date', 'Cod- SKU':'SKU', 'Código':'SKU', 'Produto.DS_PRODUTO':'Description', 'Descrição':'Description', 'Pedidos':'Orders', 'Qtde':'Orders'}
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
        st.error(f"Erro ao ler vendas: {e}")
        return pd.DataFrame()

# --- NOVO: CARREGAR FICHA TÉCNICA ---
@st.cache_data
def load_recipe_data(uploaded_file):
    try:
        try: df = pd.read_csv(uploaded_file, sep=',')
        except: df = pd.read_excel(uploaded_file)
        if df.shape[1] < 2: df = pd.read_csv(uploaded_file, sep=';')
        
        # Mapeamento para garantir nomes padronizados
        # O arquivo do usuário tem: Cod, Materia Prima, Composição (mg), Tipo
        rename_map = {
            'Cod': 'SKU', 
            'Materia Prima': 'Ingredient', 
            'Composição (mg)': 'Weight_g', # Assumindo gramas baseado no contexto (250 para 250g)
            'Tipo': 'Type'
        }
        df = df.rename(columns=rename_map)
        
        # Filtra colunas essenciais
        required_cols = ['SKU', 'Ingredient', 'Weight_g', 'Type']
        # Verifica se as colunas existem (ou se os nomes originais estão lá)
        available_cols = [c for c in required_cols if c in df.columns]
        
        if len(available_cols) < 3:
            # Tenta mapear se não encontrou pelo rename direto (flexibilidade)
            df.columns = df.columns.str.strip()
            # Reinicia rename se falhou
            return df # Retorna raw para debug se falhar
            
        df['Weight_g'] = pd.to_numeric(df['Weight_g'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Erro ao ler Ficha Técnica: {e}")
        return pd.DataFrame()

# ==============================================================================
# 2. ENGENHARIA DE FEATURES E PREVISÃO
# ==============================================================================

def filter_history_vero(df):
    mask_vero = df['Group'] == 'Vero'
    mask_date = df['Date'] >= '2025-01-01'
    keep = (mask_vero & mask_date) | (~mask_vero)
    return df[keep].copy()

def clean_outliers(df):
    df = df.sort_values(['SKU', 'Date'])
    targets = ['Americana Bola'] 
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
    d['DayOfMonth'] = d['Date'].dt.day
    d['IsPaydayWeek'] = d['DayOfMonth'].between(5, 10).astype(int)
    d['lag_1'] = d.groupby('SKU')['Orders'].shift(1)
    d['lag_7'] = d.groupby('SKU')['Orders'].shift(7)
    d['lag_14'] = d.groupby('SKU')['Orders'].shift(14)
    d['roll_7'] = d.groupby('SKU')['Orders'].shift(1).rolling(7).mean()
    return d

def run_forecast(df_raw, days_ahead=7):
    df_train_base = filter_history_vero(df_raw)
    df_train_base = clean_outliers(df_train_base)
    
    last_date = df_train_base['Date'].max()
    start_date = df_train_base['Date'].min()
    future_range = pd.date_range(start_date, last_date + timedelta(days=days_ahead))
    df_dates = pd.DataFrame({'Date': future_range})
    
    weather_full = get_weather_data(start_date, df_dates['Date'].max())
    
    np.random.seed(42)
    df_dates['Temp_Avg'] = np.random.normal(25, 3, len(df_dates))
    is_summer = df_dates['Date'].dt.month.isin([1, 2, 3, 12])
    df_dates['Rain_mm'] = np.where(is_summer, np.random.exponential(8, len(df_dates)), 4)
    
    if weather_full is not None and not weather_full.empty:
        weather_full['Date'] = pd.to_datetime(weather_full['Date'])
        df_dates = pd.merge(df_dates, weather_full, on='Date', how='left', suffixes=('', '_real'))
        df_dates['Temp_Avg'] = df_dates['Temp_Avg_real'].fillna(df_dates['Temp_Avg'])
        df_dates['Rain_mm'] = df_dates['Rain_mm_real'].fillna(df_dates['Rain_mm'])
        df_dates = df_dates[['Date', 'Temp_Avg', 'Rain_mm']]
    
    weather_future = df_dates[(df_dates['Date'] > last_date) & (df_dates['Date'] <= last_date + timedelta(days=days_ahead))][['Date', 'Temp_Avg', 'Rain_mm']].copy()
    
    holidays_df = get_holidays_calendar(df_dates['Date'].min(), df_dates['Date'].max())
    df_dates = pd.merge(df_dates, holidays_df, on='Date', how='left')
    df_dates['IsHoliday'] = df_dates['IsHoliday'].fillna(0)
    df_dates['IsHighImpact'] = df_dates['IsHighImpact'].fillna(0)
    df_dates['IsHighImpactNextDay'] = df_dates['IsHighImpact'].shift(-1).fillna(0)
    
    unique_skus = df_train_base[['SKU', 'Description', 'Group']].drop_duplicates()
    unique_skus['key'] = 1
    df_dates['key'] = 1
    
    df_master = pd.merge(df_dates, unique_skus, on='key').drop('key', axis=1)
    df_master = pd.merge(df_master, df_train_base[['Date','SKU','Orders']], on=['Date','SKU'], how='left')
    
    mask_past = df_master['Date'] <= last_date
    df_master.loc[mask_past, 'Orders'] = df_master.loc[mask_past, 'Orders'].fillna(0)
    
    df_feat = generate_features(df_master)
    
    features = ['DayOfWeek', 'IsWeekend', 'IsHoliday', 'IsHighImpact', 'IsHighImpactNextDay', 'DayOfMonth', 'IsPaydayWeek', 'Temp_Avg', 'Rain_mm', 'lag_1', 'lag_7', 'roll_7']
    
    train_full = df_feat[df_feat['Date'] <= last_date].dropna()
    if train_full.empty: return pd.DataFrame(), pd.DataFrame()

    # Modelagem (Simplificada para brevidade, mantendo lógica anterior)
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, n_jobs=-1)
    model.fit(train_full[features], train_full['Orders'])
    
    preds = []
    current_df = df_master[df_master['Date'] <= last_date].copy()
    progress_bar = st.progress(0)
    
    for i in range(1, days_ahead + 1):
        next_day = last_date + timedelta(days=i)
        cols_static = ['Date','SKU','Description','Group','Temp_Avg','Rain_mm','IsHoliday','IsHighImpact','IsHighImpactNextDay']
        next_base = df_master[df_master['Date'] == next_day][cols_static].copy()
        
        temp_full = pd.concat([current_df, next_base], sort=False)
        temp_full = generate_features(temp_full)
        
        row_pred = temp_full[temp_full['Date'] == next_day].copy()
        X_test = row_pred[features].fillna(0)
        y_pred = model.predict(X_test)
        
        row_pred['Orders'] = np.maximum(np.round(y_pred, 0), 0)
        
        is_sunday = next_day.dayofweek == 6
        is_holiday_today = row_pred['IsHoliday'].values[0] == 1
        if is_sunday or is_holiday_today: row_pred['Orders'] = 0
            
        preds.append(row_pred)
        current_df = pd.concat([current_df, row_pred], sort=False)
        progress_bar.progress(i / days_ahead)
        
    return pd.concat(preds), weather_future

# ==============================================================================
# 3. INTERFACE VISUAL
# ==============================================================================

st.markdown("""
    <style>
        .title-text { text-align: center; color: #8CFF00; font-family: sans-serif; font-weight: bold; font-size: 3rem; margin-bottom: 2rem; }
        .stAppHeader { background-color: transparent; }
    </style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    try: st.image("AF-VERDUREIRA-LOGO-HORIZONTAL-07.jpg", use_container_width=True)
    except: st.warning("⚠️ Logo (JPG) não encontrada.")

st.markdown('<h1 class="title-text">PCP - Previsão & Fábrica</h1>', unsafe_allow_html=True)

# --- ÁREA DE UPLOAD ---
c_up1, c_up2 = st.columns(2)
with c_up1:
    uploaded_file = st.file_uploader("📂 1. Histórico de Vendas (CSV/Excel)", type=['csv', 'xlsx'])
with c_up2:
    uploaded_recipe = st.file_uploader("📋 2. Ficha Técnica (Opcional)", type=['csv', 'xlsx'])

# Limpeza de memória se trocar arquivo
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
            processar = st.button("🚀 Gerar Previsão de Vendas", use_container_width=True)

        if processar:
            with st.spinner("Analisando Clima Histórico e Tendências..."):
                try:
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
            
            # --- SEÇÃO 1: PREVISÃO DO TEMPO ---
            if not weather_df.empty:
                st.divider()
                st.subheader("🌤️ Clima da Semana")
                w_disp = weather_df.head(7).copy()
                w_disp['Date'] = w_disp['Date'].dt.strftime('%d/%m')
                w_disp = w_disp.rename(columns={'Date': 'Data', 'Temp_Avg': 'Temp. Média (°C)', 'Rain_mm': 'Chuva (mm)'})
                w_disp['Temp. Média (°C)'] = w_disp['Temp. Média (°C)'].map('{:.1f}'.format)
                w_disp['Chuva (mm)'] = w_disp['Chuva (mm)'].map('{:.1f}'.format)
                st.dataframe(w_disp, hide_index=True, use_container_width=True)
            
            # --- SEÇÃO 2: RESULTADO DE VENDAS ---
            st.divider()
            st.subheader("📊 Previsão de Pedidos (Unidades)")
            
            # Tabela Resumida por Grupo
            f_start = max_date + timedelta(days=1)
            f_end = max_date + timedelta(days=7)
            mask_fore = (forecast['Date'] >= f_start) & (forecast['Date'] <= f_end)
            df_view = forecast[mask_fore].groupby('Group')['Orders'].sum().reset_index().rename(columns={'Group':'Grupo', 'Orders':'Total Unidades'})
            st.dataframe(df_view, hide_index=True, use_container_width=True)
            
            # --- SEÇÃO 3: CÁLCULO DE MATÉRIA-PRIMA (NOVO!) ---
            if uploaded_recipe:
                st.divider()
                st.subheader("🏭 Necessidade de Matéria-Prima (Compras)")
                
                df_recipe = load_recipe_data(uploaded_recipe)
                
                if not df_recipe.empty:
                    # 1. Filtra Legumes (Exclusão solicitada)
                    # Verifica se coluna Type existe, senão ignora filtro
                    if 'Type' in df_recipe.columns:
                        # Filtra tudo que contiver 'legume' (case insensitive)
                        mask_legume = df_recipe['Type'].astype(str).str.contains('Legume', case=False, na=False)
                        df_recipe = df_recipe[~mask_legume]
                        st.caption(f"✅ Filtro aplicado: Ingredientes do tipo 'Legume' foram excluídos do cálculo.")
                    
                    # 2. Cruza Previsão (SKU) com Receita (SKU/Cod)
                    # Garante que chaves sejam do mesmo tipo (str)
                    forecast['SKU_Str'] = forecast['SKU'].astype(str).str.strip()
                    df_recipe['SKU_Str'] = df_recipe['SKU'].astype(str).str.strip()
                    
                    df_mrp = pd.merge(forecast[mask_fore], df_recipe, on='SKU_Str', how='inner')
                    
                    # 3. Calcula Necessidade: (Qtd Pedido * Peso Receita) / 1000 = KG
                    df_mrp['Total_Kg'] = (df_mrp['Orders'] * df_mrp['Weight_g']) / 1000
                    
                    # 4. Agrupa por Ingrediente e Data
                    df_purchasing = df_mrp.pivot_table(
                        index='Ingredient', 
                        columns='Date', 
                        values='Total_Kg', 
                        aggfunc='sum'
                    ).fillna(0)
                    
                    # Totais
                    df_purchasing['TOTAL SEMANA (Kg)'] = df_purchasing.sum(axis=1)
                    df_purchasing = df_purchasing.sort_values('TOTAL SEMANA (Kg)', ascending=False).reset_index()
                    
                    # Formata datas nas colunas
                    cols_new = []
                    for c in df_purchasing.columns:
                        if isinstance(c, pd.Timestamp): cols_new.append(c.strftime('%d/%m'))
                        else: cols_new.append(c)
                    df_purchasing.columns = cols_new
                    
                    # Exibe e Download
                    st.dataframe(df_purchasing.style.format("{:.1f}"), use_container_width=True)
                    
                    csv_mrp = df_purchasing.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Baixar Lista de Compras (Kg)", csv_mrp, "lista_compras_materiaprima.csv", "text/csv")
                else:
                    st.warning("⚠️ Não foi possível ler a Ficha Técnica. Verifique as colunas (Cod, Materia Prima, Composição).")

            # --- SEÇÃO 4: DOWNLOAD E IA ---
            st.divider()
            
            # Download Vendas Detalhado
            df_piv = forecast.pivot_table(index=['SKU', 'Description'], columns='Date', values='Orders', aggfunc='sum').reset_index()
            csv_sales = df_piv.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Previsão de Vendas (Detalhada)", csv_sales, "previsao_vendas.csv", "text/csv")

            # IA Gemini
            st.subheader("🤖 Analista IA")
            api_key = st.text_input("Insira sua Gemini API Key:", type="password")
            
            if api_key:
                client = genai.Client(api_key=api_key)
                st.info("Conectado.")
                query = st.text_area("Pergunta sobre a produção ou vendas:", key="gemini_query")
                if st.button("Consultar IA"):
                    with st.spinner("Analisando..."):
                        # Prepara contexto resumido para a IA
                        resumo_vendas = df_view.to_string()
                        prompt = f"""
                        Você é um gerente de PCP.
                        Resumo da Previsão de Vendas (Próximos 7 dias):
                        {resumo_vendas}
                        
                        Pergunta do usuário: {query}
                        """
                        try:
                            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Erro IA: {e}")