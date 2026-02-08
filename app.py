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
import unicodedata

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="PCP Verdureira - Máquina Final", layout="wide")

# ==============================================================================
# 1. FUNÇÕES AUXILIARES
# ==============================================================================

def normalize_text(text):
    """Remove acentos, espaços extras e padroniza para minúsculo."""
    if not isinstance(text, str):
        return str(text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text.lower().strip()

def get_holidays_calendar(start_date, end_date):
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
                if br_holidays.get(d_date) == "Sexta-feira Santa": is_high = 1
            if d_date in mothers_days or d_date in fathers_days:
                is_hol = 1 
                is_high = 1 
            data.append({'Date': d, 'IsHoliday': is_hol, 'IsHighImpact': is_high})
        return pd.DataFrame(data)
    except:
        d_range = pd.date_range(start_date, end_date)
        return pd.DataFrame({'Date': d_range, 'IsHoliday': 0, 'IsHighImpact': 0})

@st.cache_data(ttl=3600)
def get_weather_data(start_date, end_date, lat=-23.55, lon=-46.63):
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
    txt = normalize_text(desc)
    if 'americana bola' in txt: return 'Americana Bola'
    vero_keys = ['vero', 'primavera', 'roxa', 'mix', 'repolho', 'couve', 'rucula hg']
    if any(x in txt for x in vero_keys): return 'Vero'
    if 'mini' in txt: return 'Minis'
    match = re.search(r'(\d+)\s*g', txt)
    if match and 'insalata' in txt:
        weight = int(match.group(1))
        if weight > 100: return 'Saladas'
    legumes_keys = ['legume', 'cenoura', 'beterraba', 'abobrinha', 'batata', 'mandioca', 'mandioquinha', 'sopa', 'grao de bico', 'lentilha', 'pinhao', 'quinoa', 'milho']
    if any(x in txt for x in legumes_keys): return 'Legumes'
    saladas_keys = ['salada', 'alface', 'rucula', 'agriao', 'insalata']
    if any(x in txt for x in saladas_keys): return 'Saladas'
    return 'Outros'

def extract_weight_from_sku(text):
    match = re.search(r'(\d+)\s*[gG]', str(text))
    if match: return float(match.group(1))
    return 0.0

# ==============================================================================
# 2. CARREGAMENTO E LEITURA DE ARQUIVOS
# ==============================================================================

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
        
        # --- FIX: Agrupamento Forçado (Mini Americana, Mini Alface Insalata & Mini Alface Insalata Prima) ---
        if 'Description' in df.columns:
            df['Description'] = df['Description'].astype(str).str.replace(r'(?i)mini\s*americana\s*80\s*g', 'Mini Americana 90g', regex=True)
            df['Description'] = df['Description'].astype(str).str.replace(r'(?i)mini\s*alface\s*insalata\s*80\s*g', 'Mini Alface Insalata 90g', regex=True)
            df['Description'] = df['Description'].astype(str).str.replace(r'(?i)mini\s*alface\s*insalata\s*prima\s*80\s*g', 'Mini Alface Insalata Prima 90g', regex=True)
        # ----------------------------------------------------------------------------------------------------

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
        df['Group'] = df['Description'].apply(classify_group)
        
        # AGREGAR POR DESCRIÇÃO PARA SOMAR OS SKUS DIFERENTES
        return df.groupby(['Date', 'Description', 'Group']).agg({'Orders': 'sum', 'SKU': 'first'}).reset_index()

    except Exception as e:
        st.error(f"Erro ao ler vendas: {e}")
        return pd.DataFrame()

@st.cache_data
def load_recipe_data(uploaded_file):
    try:
        try: df = pd.read_csv(uploaded_file, sep=',')
        except: df = pd.read_excel(uploaded_file)
        if 'Cod' not in df.columns and 'SKU' not in df.columns:
             try: df = pd.read_csv(uploaded_file, sep=',', header=1)
             except: df = pd.read_excel(uploaded_file, header=1)
        if df.shape[1] < 2: df = pd.read_csv(uploaded_file, sep=';')
        
        df.columns = df.columns.str.strip()
        if 'Cod' in df.columns and 'SKU' in df.columns:
            df = df.rename(columns={'SKU': 'SKU_Original_Nome'})
        rename_map = {'Cod': 'SKU', 'Materia Prima': 'Ingredient', 'Composição (mg)': 'Weight_g', 'Tipo': 'Type'}
        df = df.rename(columns=rename_map)
        required_cols = ['SKU', 'Ingredient', 'Weight_g', 'Type']
        cols_to_keep = [c for c in required_cols if c in df.columns]
        df = df[cols_to_keep]
        
        # --- FIX: Agrupamento Forçado na Receita ---
        if 'Ingredient' in df.columns:
            df['Ingredient'] = df['Ingredient'].astype(str).str.replace(r'(?i)mini\s*americana\s*80\s*g', 'Mini Americana 90g', regex=True)
            df['Ingredient'] = df['Ingredient'].astype(str).str.replace(r'(?i)mini\s*alface\s*insalata\s*80\s*g', 'Mini Alface Insalata 90g', regex=True)
            df['Ingredient'] = df['Ingredient'].astype(str).str.replace(r'(?i)mini\s*alface\s*insalata\s*prima\s*80\s*g', 'Mini Alface Insalata Prima 90g', regex=True)
        # -------------------------------------------

        if 'Weight_g' in df.columns:
            df['Weight_g'] = pd.to_numeric(df['Weight_g'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Erro ao ler Ficha Técnica: {e}")
        return pd.DataFrame()

@st.cache_data
def load_yield_data_scenarios(uploaded_file):
    try:
        try: df = pd.read_csv(uploaded_file, sep=',')
        except: df = pd.read_excel(uploaded_file)
        if df.shape[1] < 2: df = pd.read_csv(uploaded_file, sep=';')
        df.columns = df.columns.str.strip()
        df = df[pd.to_numeric(df['Rendimento'], errors='coerce') > 0]
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        
        # --- FIX: Agrupamento Forçado no Rendimento ---
        if 'Produto' in df.columns:
            df['Produto'] = df['Produto'].astype(str).str.replace(r'(?i)mini\s*americana\s*80\s*g', 'Mini Americana 90g', regex=True)
            df['Produto'] = df['Produto'].astype(str).str.replace(r'(?i)mini\s*alface\s*insalata\s*80\s*g', 'Mini Alface Insalata 90g', regex=True)
            df['Produto'] = df['Produto'].astype(str).str.replace(r'(?i)mini\s*alface\s*insalata\s*prima\s*80\s*g', 'Mini Alface Insalata Prima 90g', regex=True)
        # ----------------------------------------------

        df['Produto'] = df['Produto'].apply(normalize_text)
        df['Fornecedor'] = df['Fornecedor'].astype(str).str.strip().str.upper()
        df['Origem'] = np.where(df['Fornecedor'] == 'VERDE PRIMA', 'VP', 'MERCADO')
        df = df.sort_values(['Produto', 'Origem', 'Data'], ascending=[True, True, False])
        
        results = []
        for (prod, origem), group in df.groupby(['Produto', 'Origem']):
            val_1 = group['Rendimento'].iloc[0] 
            val_3 = group['Rendimento'].head(3).mean() 
            val_5 = group['Rendimento'].head(5).mean() 
            results.append({'Produto': prod, 'Origem': origem, 'Reativo (1)': val_1, 'Equilibrado (3)': val_3, 'Conservador (5)': val_5})
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Erro ao ler Rendimento: {e}")
        return pd.DataFrame()

@st.cache_data
def load_availability_data(uploaded_file):
    try:
        df = None
        try: 
            temp = pd.read_csv(uploaded_file)
            if 'Hortaliça' in temp.columns: df = temp
            else:
                temp = pd.read_csv(uploaded_file, header=1)
                if 'Hortaliça' in temp.columns: df = temp
        except: 
            try: 
                temp = pd.read_excel(uploaded_file)
                if 'Hortaliça' in temp.columns: df = temp
                else: df = pd.read_excel(uploaded_file, header=1)
            except: pass
        if df is None: return pd.DataFrame()

        df.columns = df.columns.str.strip()
        name_map = {
            'crespa verde': 'alface crespa',
            'frizzy roxa': 'frisee roxa',
            'lollo': 'lollo rossa',
            'chicoria': 'frisee chicoria',
            'barlach': 'barlach', 
        }
        if 'Hortaliça' in df.columns:
            # --- FIX: Agrupamento Forçado na Disponibilidade ---
            df['Hortaliça'] = df['Hortaliça'].astype(str).str.replace(r'(?i)mini\s*americana\s*80\s*g', 'Mini Americana 90g', regex=True)
            df['Hortaliça'] = df['Hortaliça'].astype(str).str.replace(r'(?i)mini\s*alface\s*insalata\s*80\s*g', 'Mini Alface Insalata 90g', regex=True)
            df['Hortaliça'] = df['Hortaliça'].astype(str).str.replace(r'(?i)mini\s*alface\s*insalata\s*prima\s*80\s*g', 'Mini Alface Insalata Prima 90g', regex=True)
            # ---------------------------------------------------
            
            df = df.dropna(subset=['Hortaliça'])
            df['Hortaliça_Norm'] = df['Hortaliça'].apply(normalize_text)
            def translate_name(name): return name_map.get(name, name) 
            df['Hortaliça_Traduzida'] = df['Hortaliça_Norm'].apply(translate_name)
            cols_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta']
            cols_existentes = []
            for c in df.columns:
                if c.strip() in cols_dias: cols_existentes.append(c)
            rename_dict = {c: c.strip() for c in cols_existentes}
            df = df.rename(columns=rename_dict)
            cols_clean = list(rename_dict.values())
            if cols_clean:
                return df.groupby('Hortaliça_Traduzida')[cols_clean].sum().reset_index()
            else: return pd.DataFrame()
        else: return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao ler Disponibilidade VP: {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. MOTOR DE CÁLCULO E PREVISÃO
# ==============================================================================

def treat_data_interruption(df):
    df_clean = df.copy()
    start_anomaly = pd.Timestamp('2025-02-01')
    end_anomaly = pd.Timestamp('2025-08-31')
    affected_groups = ['Saladas', 'Minis', 'Legumes']
    mask_anomaly = ((df_clean['Date'] >= start_anomaly) & (df_clean['Date'] <= end_anomaly) & (df_clean['Group'].isin(affected_groups)))
    if mask_anomaly.any():
        df_clean.loc[mask_anomaly, 'Orders'] = np.nan
        df_clean = df_clean.sort_values(['SKU', 'Date'])
        df_clean = df_clean.set_index('Date') 
        df_clean['Orders'] = df_clean.groupby('SKU')['Orders'].transform(lambda x: x.interpolate(method='time', limit_direction='both'))
        df_clean = df_clean.reset_index()
        df_clean['Orders'] = df_clean['Orders'].fillna(0)
    return df_clean

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

def calculate_backtest_accuracy(df_raw):
    try:
        df_treated = treat_data_interruption(df_raw)
        df_clean = filter_history_vero(df_treated)
        df_clean = clean_outliers(df_clean)
        last_date = df_clean['Date'].max()
        start_test = last_date - timedelta(days=6)
        train = df_clean[df_clean['Date'] < start_test]
        test = df_clean[df_clean['Date'] >= start_test]
        if train.empty or test.empty: return None, None, None
        unique_skus = df_clean[['SKU', 'Description', 'Group']].drop_duplicates()
        unique_skus['key'] = 1
        date_range_full = pd.date_range(train['Date'].min(), last_date)
        df_dates_full = pd.DataFrame({'Date': date_range_full})
        df_dates_full['key'] = 1
        df_master = pd.merge(df_dates_full, unique_skus, on='key').drop('key', axis=1)
        df_master = pd.merge(df_master, df_clean[['Date','SKU','Orders']], on=['Date','SKU'], how='left').fillna(0)
        holidays_df = get_holidays_calendar(df_master['Date'].min(), df_master['Date'].max())
        df_master = pd.merge(df_master, holidays_df, on='Date', how='left')
        df_master['IsHoliday'] = df_master['IsHoliday'].fillna(0)
        df_master['IsHighImpact'] = df_master['IsHighImpact'].fillna(0)
        df_master['IsHighImpactNextDay'] = df_master['IsHighImpact'].shift(-1).fillna(0)
        weather = get_weather_data(df_master['Date'].min(), df_master['Date'].max())
        if weather is not None: df_master = pd.merge(df_master, weather, on='Date', how='left')
        else:
             df_master['Temp_Avg'] = 25
             df_master['Rain_mm'] = 5
        df_master['Temp_Avg'] = df_master['Temp_Avg'].fillna(25)
        df_master['Rain_mm'] = df_master['Rain_mm'].fillna(0)
        df_feat = generate_features(df_master)
        features = ['DayOfWeek', 'IsWeekend', 'IsHoliday', 'IsHighImpact', 'IsHighImpactNextDay', 'DayOfMonth', 'IsPaydayWeek', 'Temp_Avg', 'Rain_mm', 'lag_1', 'lag_7', 'roll_7']
        X_train = df_feat[df_feat['Date'] < start_test].dropna()
        X_test = df_feat[df_feat['Date'] >= start_test].dropna() 
        if X_train.empty or X_test.empty: return None, None, None
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, n_jobs=-1, random_state=42)
        model.fit(X_train[features], X_train['Orders'])
        preds = model.predict(X_test[features])
        preds = np.maximum(np.round(preds, 0), 0)
        actuals = X_test['Orders'].values
        sum_abs_error = np.sum(np.abs(actuals - preds))
        sum_actuals = np.sum(actuals)
        wape = sum_abs_error / sum_actuals if sum_actuals > 0 else 0
        accuracy = max(0, 1 - wape)
        return accuracy, sum_actuals, np.sum(preds)
    except Exception as e:
        print(f"Erro Backtest: {e}")
        return None, None, None

def run_forecast(df_raw, days_ahead=8):
    df_treated = treat_data_interruption(df_raw)
    df_train_base = filter_history_vero(df_treated)
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
                m = XGBRegressor(n_estimators=cfg['n_estimators'], learning_rate=cfg['learning_rate'], max_depth=cfg['max_depth'], n_jobs=-1)
                m.fit(X_train_sub[features], X_train_sub['Orders'])
                val_preds = m.predict(X_val_sub[features])
                mae = mean_absolute_error(X_val_sub['Orders'], val_preds)
                if mae < best_mae:
                    best_mae = mae
                    best_params = cfg
            except: continue
    model = XGBRegressor(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'], n_jobs=-1)
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

c_up1, c_up2 = st.columns(2)
with c_up1:
    uploaded_file = st.file_uploader("📂 1. Histórico Vendas", type=['csv', 'xlsx'])
    uploaded_recipe = st.file_uploader("📋 2. Ficha Técnica", type=['csv', 'xlsx'])
with c_up2:
    uploaded_yield = st.file_uploader("🚜 3. Rendimento", type=['csv', 'xlsx'])
    uploaded_avail = st.file_uploader("🌾 4. Disponibilidade VP", type=['csv', 'xlsx'])

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
            processar = st.button("🚀 Gerar Planejamento Completo", use_container_width=True)

        if processar:
            with st.spinner("Calculando precisão do modelo (Backtest 7 dias)..."):
                acc, real_vol, pred_vol = calculate_backtest_accuracy(df_raw)
                st.session_state['accuracy_metric'] = (acc, real_vol, pred_vol)

            with st.spinner("Otimizando modelo (Auto-ML) e Analisando Dados..."):
                try:
                    days_horizon = 8
                    forecast_result, weather_result = run_forecast(df_raw, days_ahead=days_horizon)
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
            
            if not weather_df.empty:
                st.divider()
                st.subheader("🌤️ Clima da Semana")
                w_disp = weather_df.head(8).copy()
                w_disp['Date'] = w_disp['Date'].dt.strftime('%d/%m')
                w_disp = w_disp.rename(columns={'Date': 'Data', 'Temp_Avg': 'Temp. Média (°C)', 'Rain_mm': 'Chuva (mm)'})
                w_disp['Temp. Média (°C)'] = w_disp['Temp. Média (°C)'].map('{:.1f}'.format)
                w_disp['Chuva (mm)'] = w_disp['Chuva (mm)'].map('{:.1f}'.format)
                st.dataframe(w_disp, hide_index=True, use_container_width=True)
            
            st.divider()
            acc_data = st.session_state.get('accuracy_metric', (None, None, None))
            if acc_data[0] is not None:
                acuracia, vol_real, vol_prev = acc_data
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Acuracidade Global (7d)", f"{acuracia*100:.1f}%", help="Baseado no WAPE dos últimos 7 dias")
                kpi2.metric("Volume Real (7d)", f"{vol_real:,.0f} un")
                kpi3.metric("Volume Previsto (Backtest)", f"{vol_prev:,.0f} un", delta=f"{(vol_prev-vol_real):,.0f} un")
            
            st.subheader("📊 Resumo Executivo")
            days_summary = 8
            f_start = max_date + timedelta(days=1)
            f_end = max_date + timedelta(days=days_summary)
            ly_start = f_start - timedelta(weeks=52)
            ly_end = f_end - timedelta(weeks=52)
            l2y_start = f_start - timedelta(weeks=104)
            l2y_end = f_end - timedelta(weeks=104)
            str_ly = f"2025 ({ly_start.strftime('%d/%m')} a {ly_end.strftime('%d/%m')})"
            str_2y = f"2024 ({l2y_start.strftime('%d/%m')} a {l2y_end.strftime('%d/%m')})"
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
                summary.append({'Grupo': g, 'Previsão 8d': int(v_curr), str_ly: int(v_ly), 'Var % (vs 25)': f"{p_ly:+.1f}%", str_2y: int(v_2y), 'Var % (vs 24)': f"{p_2y:+.1f}%"})
            tot_cur = forecast['Orders'].sum()
            tot_ly = hist_ly['Orders'].sum()
            tot_2y = hist_2y['Orders'].sum()
            pt_ly = ((tot_cur / tot_ly) - 1) * 100 if tot_ly > 0 else 0
            pt_2y = ((tot_cur / tot_2y) - 1) * 100 if tot_2y > 0 else 0
            summary.append({'Grupo': 'TOTAL GERAL', 'Previsão 8d': int(tot_cur), str_ly: int(tot_ly), 'Var % (vs 25)': f"{pt_ly:+.1f}%", str_2y: int(tot_2y), 'Var % (vs 24)': f"{pt_2y:+.1f}%"})
            df_summary = pd.DataFrame(summary)
            st.dataframe(df_summary, hide_index=True, use_container_width=True)
            st.caption(f"ℹ️ As datas de comparação seguem a lógica de 'Semana Comercial' ({days_summary} dias).")
            
            st.divider()
            st.write("### 🗓️ Previsão Detalhada (Vendas)")
            df_piv = forecast.pivot_table(index=['SKU', 'Description', 'Group'], columns='Date', values='Orders', aggfunc='sum').reset_index()
            cols_new = []
            for c in df_piv.columns:
                if isinstance(c, pd.Timestamp): cols_new.append(c.strftime('%d/%m'))
                else: cols_new.append(c)
            df_piv.columns = cols_new
            st.dataframe(df_piv, use_container_width=True)
            csv_sales = df_piv.to_csv(index=False).encode('utf-8')
            st.download_button("📥 1. Baixar Previsão de Vendas (CSV)", csv_sales, "previsao_vendas.csv", "text/csv")

            if uploaded_recipe:
                st.divider()
                st.subheader("🏭 Planejamento de Compras")
                df_recipe = load_recipe_data(uploaded_recipe)
                if not df_recipe.empty:
                    df_check = df_recipe.copy()
                    df_check['Label_Weight'] = df_check['SKU'].apply(extract_weight_from_sku)
                    df_check_grouped = df_check.groupby(['SKU', 'Label_Weight'])['Weight_g'].sum().reset_index()
                    df_check_grouped = df_check_grouped[df_check_grouped['Label_Weight'] > 0]
                    df_check_grouped['Diff_Pct'] = ((df_check_grouped['Weight_g'] - df_check_grouped['Label_Weight']) / df_check_grouped['Label_Weight']) * 100
                    alerts = df_check_grouped[df_check_grouped['Diff_Pct'] > 5.0].sort_values('Diff_Pct', ascending=False)
                    if not alerts.empty:
                        with st.expander("⚠️ Alerta: Discrepâncias de Peso (>5%)", expanded=True):
                            for index, row in alerts.iterrows(): st.warning(f"🔴 **{row['SKU']}**: Rótulo {row['Label_Weight']:.0f}g vs Receita {row['Weight_g']:.0f}g (+{row['Diff_Pct']:.1f}%)")
                    
                    if 'Type' in df_recipe.columns:
                        mask_legume = df_recipe['Type'].astype(str).str.contains('Legume', case=False, na=False)
                        df_recipe = df_recipe[~mask_legume]
                    
                    forecast['SKU_Str'] = forecast['SKU'].astype(str).str.strip()
                    df_recipe['SKU_Str'] = df_recipe['SKU'].astype(str).str.strip()
                    mask_fore = (forecast['Date'] >= f_start) & (forecast['Date'] <= f_end)
                    df_mrp = pd.merge(forecast[mask_fore], df_recipe, on='SKU_Str', how='inner')
                    df_mrp['Total_Kg'] = (df_mrp['Orders'] * df_mrp['Weight_g']) / 1000
                    
                    def check_rigid(row):
                        ing = normalize_text(str(row['Ingredient']))
                        desc = normalize_text(str(row['Description']))
                        return ing in desc

                    df_mrp['Is_Rigid'] = df_mrp.apply(check_rigid, axis=1)
                    df_mrp['Date_Clean'] = df_mrp['Date'] 
                    df_mrp['DayNum'] = df_mrp['Date'].dt.dayofweek
                    mask_sat = df_mrp['DayNum'] == 5
                    df_mrp.loc[mask_sat, 'Date'] = df_mrp.loc[mask_sat, 'Date'] - timedelta(days=1)
                    df_mrp['Ingredient_Norm'] = df_mrp['Ingredient'].apply(normalize_text) 
                    df_kg_daily = df_mrp.groupby(['Ingredient_Norm', 'Ingredient', 'Date', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
                    if True not in df_kg_daily.columns: df_kg_daily[True] = 0
                    if False not in df_kg_daily.columns: df_kg_daily[False] = 0
                    df_kg_daily = df_kg_daily.rename(columns={True: 'Demand_Rigid', False: 'Demand_Flex'})
                    df_kg_daily['Total_Kg'] = df_kg_daily['Demand_Rigid'] + df_kg_daily['Demand_Flex']
                    
                    if uploaded_avail and uploaded_yield:
                        df_avail = load_availability_data(uploaded_avail)
                        df_yield_scenarios = load_yield_data_scenarios(uploaded_yield)
                        if not df_avail.empty and not df_yield_scenarios.empty:
                            st.write("#### 🎯 Calibragem de Rendimento")
                            col_sel1, col_sel2 = st.columns([1, 2])
                            with col_sel1:
                                scenario = st.radio("Escolha o perfil de risco do rendimento:", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1)
                            map_dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
                            df_kg_daily['DayNum'] = df_kg_daily['Date'].dt.dayofweek
                            df_kg_daily['DayName'] = df_kg_daily['DayNum'].map(map_dias)
                            id_vars = ['Hortaliça_Traduzida']
                            val_vars = [c for c in df_avail.columns if c in map_dias.values()]
                            df_avail_melt = df_avail.melt(id_vars=id_vars, value_vars=val_vars, var_name='DayName', value_name='Kg_Available')
                            df_proc = pd.merge(df_kg_daily, df_avail_melt, left_on=['Ingredient_Norm', 'DayName'], right_on=['Hortaliça_Traduzida', 'DayName'], how='left')
                            df_proc['Kg_Available'] = df_proc['Kg_Available'].fillna(0)
                            today = pd.Timestamp.now().normalize()
                            df_proc = df_proc[df_proc['Date'] > today]
                            groups_sub = {'Frutas Verdes': ['alface crespa', 'escarola', 'frisee chicoria', 'lalique', 'romana'], 'Frutas Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa', 'barlach']}
                            log_substitutions = []
                            df_final_rows = []
                            for day, group_day in df_proc.groupby('Date'):
                                group_day['Used_VP_Rigid'] = np.minimum(group_day['Kg_Available'], group_day['Demand_Rigid'])
                                group_day['Deficit_Rigid'] = group_day['Demand_Rigid'] - group_day['Used_VP_Rigid']
                                group_day['Remaining_VP'] = group_day['Kg_Available'] - group_day['Used_VP_Rigid']
                                group_day['Balance_Flex'] = group_day['Remaining_VP'] - group_day['Demand_Flex']
                                for g_name, items in groups_sub.items():
                                    mask_g = group_day['Ingredient_Norm'].isin(items)
                                    df_g = group_day[mask_g].copy()
                                    if not df_g.empty:
                                        surplus = df_g[df_g['Balance_Flex'] > 0]
                                        deficit = df_g[df_g['Balance_Flex'] < 0]
                                        if not surplus.empty and not deficit.empty:
                                            pool_surplus = surplus[['Ingredient', 'Balance_Flex']].to_dict('records')
                                            for idx, row_def in deficit.iterrows():
                                                needed = abs(row_def['Balance_Flex'])
                                                for src in pool_surplus:
                                                    if src['Balance_Flex'] > 0 and needed > 0:
                                                        transfer = min(src['Balance_Flex'], needed)
                                                        log_substitutions.append({'Data': day.strftime('%d/%m'), 'Grupo': g_name, 'Falta (Flex)': row_def['Ingredient'], 'Substituto': src['Ingredient'], 'Qtd (Kg)': transfer})
                                                        src['Balance_Flex'] -= transfer
                                                        needed -= transfer
                                                        idx_src = group_day[group_day['Ingredient'] == src['Ingredient']].index[0]
                                                        group_day.at[idx, 'Remaining_VP'] += transfer
                                                        group_day.at[idx_src, 'Demand_Flex'] += transfer
                                                        group_day.at[idx, 'Demand_Flex'] -= transfer
                                df_final_rows.append(group_day)
                            if df_final_rows:
                                df_processed = pd.concat(df_final_rows)
                                df_processed['Used_VP_Flex'] = np.minimum(df_processed['Remaining_VP'], df_processed['Demand_Flex'])
                                df_processed['Deficit_Flex'] = df_processed['Demand_Flex'] - df_processed['Used_VP_Flex']
                                df_processed['Kg_VP'] = df_processed['Used_VP_Rigid'] + df_processed['Used_VP_Flex']
                                df_processed['Kg_Mkt'] = df_processed['Deficit_Rigid'] + df_processed['Deficit_Flex']
                                df_processed['Sobra_VP'] = df_processed['Remaining_VP'] - df_processed['Used_VP_Flex']
                                df_processed['Sobra_VP'] = df_processed['Sobra_VP'].clip(lower=0)
                                df_final = pd.merge(df_processed, df_yield_scenarios, left_on='Ingredient_Norm', right_on='Produto', how='left')
                                col_yield = scenario
                                df_y_pivot = df_yield_scenarios.pivot(index='Produto', columns='Origem', values=col_yield).reset_index()
                                df_y_pivot.columns.name = None
                                df_y_pivot = df_y_pivot.rename(columns={'VP': 'Y_VP', 'MERCADO': 'Y_MKT'})
                                df_calc = pd.merge(df_processed, df_y_pivot, left_on='Ingredient_Norm', right_on='Produto', how='left')
                                df_calc['Y_VP'] = df_calc['Y_VP'].fillna(10.0)
                                df_calc['Y_MKT'] = df_calc['Y_MKT'].fillna(10.0)
                                df_calc['Boxes_Mkt'] = np.ceil(df_calc['Kg_Mkt'] / df_calc['Y_MKT'])
                                st.write(f"#### 🛒 Ordem de Compra Diária (Caixas Mercado - Cenário {scenario})")
                                df_daily_view = df_calc.pivot_table(index='Ingredient', columns='Date', values='Boxes_Mkt', aggfunc='sum').fillna(0)
                                cols_fmt = []
                                for c in df_daily_view.columns:
                                    d_str = c.strftime('%d/%m (%a)')
                                    d_str = d_str.replace('Mon', 'Seg').replace('Tue', 'Ter').replace('Wed', 'Qua').replace('Thu', 'Qui').replace('Fri', 'Sex')
                                    cols_fmt.append(d_str)
                                df_daily_view.columns = cols_fmt
                                df_daily_view['TOTAL PERÍODO'] = df_daily_view.sum(axis=1)
                                df_daily_view = df_daily_view[df_daily_view['TOTAL PERÍODO'] > 0]
                                st.dataframe(df_daily_view.style.format("{:.0f}"), use_container_width=True)
                                if log_substitutions:
                                    with st.expander("🔄 Relatório de Substituições (Apenas Demanda Flexível)", expanded=False): st.dataframe(pd.DataFrame(log_substitutions))
                                with st.expander("📊 Detalhes de Rendimento Utilizado (Auditoria)", expanded=False):
                                    audit_yield = df_calc[['Ingredient', 'Y_VP', 'Y_MKT']].drop_duplicates().dropna()
                                    audit_yield = audit_yield.rename(columns={'Y_VP': 'Rendimento VP (Kg/Cx)', 'Y_MKT': 'Rendimento Mkt (Kg/Cx)'})
                                    num_yield = audit_yield.select_dtypes(include=[np.number]).columns
                                    st.dataframe(audit_yield.style.format("{:.2f}", subset=num_yield))
                                with st.expander("🔍 Diagnóstico de Nomes e Match (Debug)", expanded=False):
                                    vp_items = set(df_avail['Hortaliça_Traduzida'].unique())
                                    dem_items = set(df_kg_daily['Ingredient_Norm'].unique())
                                    missing_in_demand = vp_items - dem_items
                                    if missing_in_demand: st.warning(f"Itens na VP sem match na Fábrica: {', '.join(missing_in_demand)}")
                                    st.write("--- Raio-X Rúcula ---")
                                    df_rucula = df_calc[df_calc['Ingredient_Norm'].str.contains('rucula', na=False)]
                                    if not df_rucula.empty:
                                        cols_rucula = ['Date', 'Ingredient', 'Total_Kg', 'Kg_Available', 'Sobra_VP', 'Kg_Mkt']
                                        st.dataframe(df_rucula[cols_rucula].style.format("{:.1f}", subset=['Total_Kg', 'Kg_Available', 'Sobra_VP', 'Kg_Mkt']))
                                    else: st.write("Nenhuma linha de 'rucula' encontrada no cálculo final.")
                                df_surplus_daily = df_calc[df_calc['Sobra_VP'] > 0]
                                if not df_surplus_daily.empty:
                                    with st.expander(f"🚜 Sobras Verde Prima (Visão Diária)", expanded=False):
                                        df_surplus_view = df_surplus_daily.pivot_table(index='Ingredient', columns='Date', values='Sobra_VP', aggfunc='sum').fillna(0)
                                        cols_s_fmt = []
                                        for c in df_surplus_view.columns:
                                            d_str = c.strftime('%d/%m (%a)')
                                            d_str = d_str.replace('Mon', 'Seg').replace('Tue', 'Ter').replace('Wed', 'Qua').replace('Thu', 'Qui').replace('Fri', 'Sex')
                                            cols_s_fmt.append(d_str)
                                        df_surplus_view.columns = cols_s_fmt
                                        st.dataframe(df_surplus_view.style.format("{:.1f}"), use_container_width=True)
                                csv_order = df_daily_view.to_csv().encode('utf-8')
                                st.download_button("📥 Baixar Ordem de Compra (Diária)", csv_order, "ordem_compra_diaria.csv", "text/csv")
                            else: st.info("Sem demandas futuras para processar.")
                        else: st.warning("⚠️ Arquivos auxiliares inválidos ou vazios.")
                    else:
                        st.info("💡 Suba Rendimento e Disponibilidade para cálculo de caixas.")
                        df_kg_show = df_kg_daily.pivot_table(index='Ingredient', columns='Date', values='Total_Kg', aggfunc='sum').fillna(0)
                        st.dataframe(df_kg_show.style.format("{:.1f}"), use_container_width=True)

            st.divider()
            st.subheader("🤖 Analista IA")
            api_key = st.text_input("Insira sua Gemini API Key:", type="password")
            if api_key:
                client = genai.Client(api_key=api_key)
                query = st.text_area("Pergunta sobre a produção ou vendas:", key="gemini_query")
                if st.button("Consultar IA"):
                    with st.spinner("Analisando..."):
                        resumo_str = df_summary.to_string()
                        prompt = f"""
                        Você é um gerente de PCP.
                        Resumo Executivo:
                        {resumo_str}
                        Pergunta: {query}
                        """
                        try:
                            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Erro IA: {e}")