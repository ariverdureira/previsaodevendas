import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays
import traceback

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
    desc = desc.lower()
    
    if 'americana bola' in desc: return 'Americana Bola'
    if any(x in desc for x in ['vero', 'primavera', 'roxa']): return 'Vero'
    if 'mini' in desc: return 'Minis'
    if any(x in desc for x in ['legume', 'cenoura', 'beterraba', 'abobrinha']): return 'Legumes'
    if any(x in desc for x in ['salada', 'alface', 'rúcula', 'agrião']): return 'Saladas'
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
        df['Group'] = df['Description'].apply(classify_group)
        
        return df.groupby(['Date','SKU','Description','Group'])['Orders'].sum().reset_index()
    except Exception as e:
        st.error(f"Erro leitura: {e}")
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
        target_groups = ['Vero', 'Americana Bola']
        skus = df[df['Group'].isin(target_groups)]['SKU'].unique()
        
        for sku in skus:
            mask = df['SKU'] == sku
            series = df.loc[mask, 'Orders']
            if len(series) < 5: continue
            
            roll_med = series.rolling(window=14, min_periods=1, center=True).median()
            is_outlier = series > (roll_med * 4)
            if is_outlier.any():
                df.loc[mask & is_outlier, 'Orders'] = roll_med[is_outlier]
        return df
    except:
        return df

# --- 4. MOTOR DE PREVISÃO ---
def run_forecast(df_hist_raw, days_ahead=14):
    try:
        # 1. Filtros
        df_hist = filter_history_vero(df_hist_raw)
        df_hist = clean_outliers(df_hist)
        
        end_date = df_hist['Date'].max()
        start_date = df_hist['Date'].min()
        dates_future = pd.date_range(start_date, end_date + timedelta(days=days_ahead))
        
        # 2. Base Clima e Feriados
        df_dates = pd.DataFrame({'Date': dates_future})
        weather = get_live_forecast(days=days_ahead)
        
        np.random.seed(42)
        df_dates['Temp_Avg'] = np.random.normal(25, 3, len(df_dates))
        is_summer = df_dates['Date'].dt.month.isin([1,2,3,12])
        df_dates['Rain_mm'] = np.where(is_summer, np.random.exponential(8, len(df_dates)), 4)
        
        if weather is not None:
            weather['Date'] = pd.to_datetime(weather['Date'])
            df_dates = pd.merge(df_dates, weather, on='Date', how='left', suffixes=('', '_real'))
            df_dates['Temp_Avg'] = df_dates['Temp_Avg_real'].fillna(df_dates['Temp_Avg'])
            df_dates['Rain_mm'] = df_dates['Rain_mm_real'].fillna(df_dates['Rain_mm'])
            df_dates = df_dates[['Date', 'Temp_Avg', 'Rain_mm']]

        holidays_df = get_holidays_calendar(df_dates['Date'].min(), df_dates['Date'].max())
        df_dates = pd.merge(df_dates, holidays_df, on='Date', how='left')
        df_dates['IsHoliday'] = df_dates['IsHoliday'].fillna(0)

        # 3. Merge Principal
        skus = df_hist[['SKU', 'Description', 'Group']].drop_duplicates()
        skus['key'] = 1
        df_dates['key'] = 1
        
        df_master = pd.merge(df_dates, skus, on='key').drop('key', axis=1)
        
        df_master = pd.merge(
            df_master, 
            df_hist[['Date','SKU','Orders']], 
            on=['Date','SKU'], 
            how='left'
        )
        
        mask_hist = df_master['Date'] <= end_date
        df_master.loc[mask_hist, 'Orders'] = df_master.loc[mask_hist, 'Orders'].fillna(0)

        # 4. Features
        def gen_feat(d):
            d = d.sort_values(['SKU', 'Date'])
            d['DayOfWeek'] = d['Date'].dt.dayofweek
            d['IsWeekend'] = (d['DayOfWeek'] >= 5).astype(int)
            for l in [1, 7, 14]:
                d[f'lag_{l}'] = d.groupby('SKU')['Orders'].shift(l)
            d['rolling_mean_7'] = d.groupby('SKU')['Orders'].shift(1).rolling(7).mean()
            return d

        df_feat = gen_feat(df_master)
        train = df_feat[df_feat['Date'] <= end_date].dropna()
        
        if train.empty:
            st.error("Sem dados suficientes para treino.")
            return pd.DataFrame(), df_hist_raw

        # 5. Treino
        features = ['DayOfWeek','IsWeekend','IsHoliday','Temp_Avg','Rain_mm','lag_1','lag_7','rolling_mean_7']
        model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, n_jobs=-1)
        model.fit(train[features], train['Orders'])

        # 6. Previsão Recursiva
        preds = []
        curr = df_master[df_master['Date'] <= end_date].copy()
        prog_bar = st.progress(0)
        
        for i in range(1, days_ahead + 1):
            next_date = end_date + timedelta(days=i)
            
            cols = ['Date','SKU','Description','Group','Temp_Avg','Rain_mm','IsHoliday']
            next_base = df_master[df_master['Date'] == next_date][cols].copy()
            
            temp_full = pd.concat([curr, next_base], sort=False)
            temp_full = gen_feat(temp_full)
            
            row_to_predict = temp_full[temp_full['Date'] == next_date].copy()
            
            X = row_to_predict[features].fillna(0)
            y_pred = model.predict(X)
            y_pred = np.maximum(y_pred, 0)
            
            row_to_predict['Orders'] = np.round(y_pred, 0)
            
            # --- REGRAS DE NEGÓCIO ---
            is_sun = next_date.dayofweek == 6
            is_hol = row_to_predict['IsHoliday'].values[0] == 1
            
            if is_sun or is_hol:
                row_to_predict['Orders'] = 0
            
            preds.append(row_to_predict)
            curr = pd.concat([curr, row_to_predict], sort=False)
            prog_bar.progress(i / days_ahead)
            
        return pd.concat(preds), df_hist_raw
        
    except Exception as e:
        st.error(f"Erro Forecasting: {str(e)}")
        st.write(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()

# --- 5. INTERFACE ---
uploaded_file = st.file_uploader("📂 Carregue data.xlsx", type=['csv', 'xlsx'])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if not df_raw.empty:
        today = df_raw['Date'].max()
        st.info(f"Dados até: **{today.date()}**")
        
        if st.button("🚀 Gerar Previsão 14 Dias"):
            forecast, history = run_forecast(df_raw, days_ahead=14)
            
            if not forecast.empty:
                # Datas de Comparação
                f_start = today + timedelta(days=1)
                f_end = today + timedelta(days=14)
                
                ly_start = f_start - timedelta(weeks=52)
                ly_end = f_end - timedelta(weeks=52)
                
                l2y_start = f_start - timedelta(weeks=104)
                l2y_end = f_end - timedelta(weeks=104)

                # Resumo por Grupo
                groups = ['Americana Bola', 'Vero', 'Saladas', 'Legumes', 'Minis', 'Outros']
                summary_list = []

                for g in groups:
                    val_curr = forecast[forecast['Group'] == g]['Orders'].sum()
                    
                    hist_ly = history[(history['Date'] >= ly_start) & (history['Date'] <= ly_end)]
                    val_ly = hist_ly[hist_ly['Group'] == g]['Orders'].sum()
                    
                    hist_2y = history[(history['Date'] >= l2y_start) & (history['Date'] <= l2y_end)]
                    val_2y = hist_2y[hist_2y['Group'] == g]['Orders'].sum()
                    
                    diff_ly = ((val_curr / val_ly) - 1) * 100 if val_ly > 0 else 0
                    diff_2y = ((val_curr / val_2y) - 1) * 100 if val_2y > 0 else 0
                    
                    summary_list.append({
                        'Grupo': g,
                        'Prev 14d': int(val_curr),
                        '2025': int(val_ly),
                        'Var % (25)': f"{diff_ly:+.1f}%",
                        '2024': int(val_2y),
                        'Var % (24)': f"{diff_2y:+.1f}%"
                    })

                df_sum = pd.DataFrame(summary_list)
                
                st.divider()
                st.subheader("📊 Comparativo de Performance")
                st.dataframe(df_sum, hide_index=True, use_container_width=True)
                
                # Tabela Pivotada
                df_pivot = forecast.pivot_table(
                    index=['SKU', 'Description', 'Group'], 
                    columns='Date', 
                    values='Orders', 
                    aggfunc='sum'
                ).reset_index()
                
                cols_fmt = []
                for c in df_pivot.columns:
                    if isinstance(c, pd.Timestamp):
                        cols_fmt.append(c.strftime('%d/%m'))
                    else:
                        cols_fmt.append(c)
                df_pivot.columns = cols_fmt
                
                st.write("### 🗓️ Previsão Detalhada")
                st.dataframe(df_pivot)
                
                csv = df_pivot.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Baixar Planilha", csv, "previsao_final.csv", "text/csv")