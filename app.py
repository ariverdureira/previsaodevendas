import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays
import traceback

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="PCP Verdureira - Eficiência em Perecíveis", layout="wide")

# ==============================================================================
# 1. MOTOR DE INTELIGÊNCIA (CONTEXTO DE DEMANDA)
# ==============================================================================

def get_smart_calendar(start_date, end_date):
    """Gera contexto para a IA: Feriados, Clima e Ciclo de Pagamento"""
    br_holidays = holidays.Brazil(subdiv='SP')
    df_cal = pd.DataFrame({'Date': pd.date_range(start_date, end_date)})
    
    # Feriados e Vizinhança Nervosa (Picos de Véspera)
    df_cal['IsHoliday'] = df_cal['Date'].apply(lambda x: 1 if x in br_holidays else 0)
    df_cal['Holiday_Eve'] = df_cal['IsHoliday'].shift(-1).fillna(0)
    df_cal['Holiday_Eve_2'] = df_cal['IsHoliday'].shift(-2).fillna(0)
    
    # Datas Especiais (Dia das Mães/Pais)
    def check_special(d):
        if (d.month == 5 or d.month == 8) and d.weekday() == 6:
            if 7 < d.day <= 14: return 1
        return 0
    df_cal['IsSpecialEvent'] = df_cal['Date'].apply(check_special)
    df_cal['Special_Eve'] = df_cal['IsSpecialEvent'].shift(-1).fillna(0) | df_cal['IsSpecialEvent'].shift(-2).fillna(0)

    # Ciclo de Pagamento (Pico de demanda de supermercado)
    df_cal['Is_Payday_Week'] = df_cal['Date'].dt.day.between(5, 12).astype(int)
    
    # Clima (Influencia direta no consumo de saladas)
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": -23.55, "longitude": -46.63, "daily": ["temperature_2m_max", "precipitation_sum"],
                  "timezone": "America/Sao_Paulo", "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
        r = requests.get(url, params=params, timeout=5).json()
        df_w = pd.DataFrame({'Date': pd.to_datetime(r['daily']['time']), 'Temp_Max': r['daily']['temperature_2m_max'], 'Rain_mm': r['daily']['precipitation_sum']})
        df_cal = pd.merge(df_cal, df_w, on='Date', how='left')
    except:
        df_cal['Temp_Max'], df_cal['Rain_mm'] = 25.0, 0.0
        
    return df_cal.fillna(0)

# ==============================================================================
# 2. MOTOR DE PREVISÃO (XGBOOST - O CÉREBRO)
# ==============================================================================

def run_lean_forecast(df_v):
    df = df_v.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    def classify(desc):
        txt = str(desc).lower()
        if 'americana' in txt: return 'Americana Bola'
        if any(x in txt for x in ['vero', 'primavera', 'roxa']): return 'Vero'
        return 'Saladas'
    df['Group'] = df['Description'].apply(classify)
    
    # Novo Normal Vero (Base de aprendizado curta)
    mask_vero = (df['Group'] == 'Vero') & (df['Date'] >= '2025-01-01')
    mask_others = (df['Group'] != 'Vero')
    df_train = df[mask_vero | mask_others].copy()
    
    last_date = df['Date'].max()
    df_cal = get_smart_calendar(df_train['Date'].min(), last_date + timedelta(days=7))
    df_train = pd.merge(df_train, df_cal, on='Date', how='left')
    
    # Features Inteligentes
    df_train['DayOfWeek'] = df_train['Date'].dt.dayofweek
    df_train['lag_7'] = df_train.groupby('SKU')['Orders'].shift(7)
    df_train['lag_14'] = df_train.groupby('SKU')['Orders'].shift(14)
    
    features = ['DayOfWeek', 'lag_7', 'lag_14', 'IsHoliday', 'Holiday_Eve', 'Is_Payday_Week', 'Temp_Max']
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
    train_clean = df_train.dropna(subset=['lag_7', 'lag_14'])
    model.fit(train_clean[features], train_clean['Orders'])
    
    # Predição D+1 em diante
    future_range = pd.date_range(last_date + timedelta(days=1), last_date + timedelta(days=7))
    unique_skus = df[['SKU', 'Description', 'Group']].drop_duplicates()
    
    preds = []
    for d in future_range:
        temp = unique_skus.copy()
        temp['Date'] = d
        cal_dia = df_cal[df_cal['Date'] == d]
        for f in ['IsHoliday', 'Holiday_Eve', 'Is_Payday_Week', 'Temp_Max']:
            temp[f] = cal_dia[f].values[0] if not cal_dia.empty else 0
        temp['DayOfWeek'] = d.dayofweek
        
        # Buscar Lags Reais
        l7 = df[df['Date'] == (d - timedelta(days=7))][['SKU', 'Orders']].rename(columns={'Orders': 'lag_7'})
        l14 = df[df['Date'] == (d - timedelta(days=14))][['SKU', 'Orders']].rename(columns={'Orders': 'lag_14'})
        temp = pd.merge(temp, l7, on='SKU', how='left')
        temp = pd.merge(temp, l14, on='SKU', how='left').fillna(0)
        
        # Previsão Líquida (Sem Buffer)
        temp['Orders'] = np.maximum(0, np.round(model.predict(temp[features])))
        if d.dayofweek == 6 or temp['IsHoliday'].iloc[0] == 1: temp['Orders'] = 0
        preds.append(temp)
        
    return pd.concat(preds), df_train

# ==============================================================================
# 3. INTERFACE E REGRAS PCP (FOCADO EM EFICIÊNCIA)
# ==============================================================================

st.title("🌱 PCP Verdureira - Gestão de Perecíveis Just-in-Time")

u1, u2 = st.columns(2)
with u1:
    f_vendas = st.file_uploader("1. Vendas", type=['xlsx', 'csv'])
    f_ficha = st.file_uploader("2. Ficha Técnica", type=['xlsx', 'csv'])
with u2:
    f_rend = st.file_uploader("3. Rendimento", type=['xlsx', 'csv'])
    f_avail = st.file_uploader("4. Disponibilidade VP (Caixas)", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    # Cargas (Deduplicadas e Auditadas)
    df_v = pd.read_excel(f_vendas) # Adicionar lógica de carga robusta aqui
    df_r = pd.read_excel(f_ficha)
    df_y = pd.read_excel(f_rend)
    df_a = pd.read_excel(f_avail, header=2)

    scenario = st.radio("Cenário de Rendimento:", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
    
    if st.button("🚀 Gerar Planejamento de Fábrica"):
        # 1. PREVISÃO LÍQUIDA
        forecast, df_hist_full = run_lean_forecast(df_v)
        
        # 2. MRP (Kg Necessários)
        df_r['SKU'] = df_r['SKU'].astype(str).str.strip()
        forecast['SKU'] = forecast['SKU'].astype(str).str.strip()
        mrp = pd.merge(forecast, df_r, on='SKU', how='inner')
        mrp['Total_Kg'] = (mrp['Orders'] * pd.to_numeric(mrp['Weight_g'])) / 1000

        # REGRA: Rigidez (Não substitui se ingrediente estiver no nome do SKU)
        mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)
        
        # REGRA: Sábado antecipado para Sexta
        mrp['Date_PCP'] = mrp['Date']
        mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date_PCP'] = mrp['Date'] - timedelta(days=1)
        
        need_daily = mrp.groupby(['Date_PCP', 'Ingredient', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
        need_daily = need_daily.rename(columns={True: 'Demanda_Rigida', False: 'Demanda_Flexivel', 'Date_PCP': 'Date'})
        
        # 3. ABASTECIMENTO PRIORITÁRIO (Fazenda Própria)
        map_dias = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
        need_daily['DayName'] = need_daily['Date'].dt.dayofweek.map(map_dias)
        
        # Rendimentos por cenário
        df_y['Data'] = pd.to_datetime(df_y['Data'])
        yield_map = []
        for (prod, forn), g in df_y.groupby(['Produto', 'Fornecedor']):
            g = g.sort_values('Data', ascending=False)
            val = g['Rendimento'].iloc[0] if '1' in scenario else (g['Rendimento'].head(3).mean() if '3' in scenario else g['Rendimento'].head(5).mean())
            yield_map.append({'Produto': str(prod).lower().strip(), 'Origem': 'VP' if 'VERDE PRIMA' in str(forn).upper() else 'MKT', 'Y_Val': val})
        df_y_f = pd.DataFrame(yield_map)

        # Disponibilidade (Caixas -> Kg)
        df_a['Ing_Key'] = df_a['Hortaliça'].str.lower().str.strip()
        avail_melt = df_a.melt(id_vars='Ing_Key', var_name='DayName', value_name='Boxes_VP').fillna(0)
        y_vp = df_y_f[df_y_f['Origem'] == 'VP'].rename(columns={'Y_Val': 'Y_VP'})
        avail_kg = pd.merge(avail_melt, y_vp, left_on='Ing_Key', right_on='Produto', how='left')
        avail_kg['Kg_VP'] = pd.to_numeric(avail_kg['Boxes_VP'], errors='coerce').fillna(0) * avail_kg['Y_VP'].fillna(10.0)

        # Merge Necessidade vs Fazenda
        df_proc = pd.merge(need_daily, avail_kg[['Ing_Key', 'DayName', 'Kg_VP']], left_on=['Ingredient', 'DayName'], right_on=['Ing_Key', 'DayName'], how='left').fillna(0)
        
        # 4. SUBSTITUIÇÕES DE GRUPO (Minimizar perda no campo)
        groups_sub = {
            'Verdes': ['alface crespa', 'escarola', 'frisee chicória', 'lalique', 'romana'],
            'Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa']
        }

        final_rows = []
        for date, g in df_proc.groupby('Date'):
            # Consome o estoque da fazenda na demanda Rígida primeiro
            g['Used_VP_Rigid'] = np.minimum(g['Kg_VP'], g.get('Demanda_Rigida', 0))
            g['Sobra_VP'] = g['Kg_VP'] - g['Used_VP_Rigid']
            # O que sobrou atende a demanda Flexível do próprio item
            g['Used_VP_Flex'] = np.minimum(g['Sobra_VP'], g.get('Demanda_Flexivel', 0))
            g['Sobra_Item'] = g['Sobra_VP'] - g['Used_VP_Flex']
            g['Deficit_Flex'] = g.get('Demanda_Flexivel', 0) - g['Used_VP_Flex']
            
            # Substituição: Sobra de um atende falta de outro do mesmo grupo
            for g_name, members in groups_sub.items():
                mask = g['Ingredient'].str.lower().str.strip().isin(members)
                if mask.any():
                    pool_sobra = g.loc[mask, 'Sobra_Item'].sum()
                    pool_falta = g.loc[mask, 'Deficit_Flex'].sum()
                    if pool_sobra > 0 and pool_falta > 0:
                        compensa = min(pool_sobra, pool_falta)
                        ratio = (pool_falta - compensa) / pool_falta if pool_falta > 0 else 0
                        g.loc[mask, 'Deficit_Flex'] *= ratio
            
            # Déficit final que gera Ordem de Compra
            g['Deficit_Líquido_Kg'] = (g.get('Demanda_Rigida', 0) - g['Used_VP_Rigid']) + g['Deficit_Flex']
            final_rows.append(g)

        df_final = pd.concat(final_rows)
        
        # 5. ORDEM DE COMPRA (MERCADO - SEM BUFFER)
        y_mkt = df_y_f[df_y_f['Origem'] == 'MKT'].groupby('Produto')['Y_Val'].mean().reset_index().rename(columns={'Y_Val': 'Y_MKT'})
        df_final['Prod_Low'] = df_final['Ingredient'].str.lower().str.strip()
        df_final = pd.merge(df_final, y_mkt, left_on='Prod_Low', right_on='Produto', how='left')
        df_final['Boxes_Buy'] = np.ceil(df_final['Deficit_Líquido_Kg'] / df_final['Y_MKT'].fillna(10.0))

        # VISUALIZAÇÃO
        st.subheader(f"🛒 Ordem de Compra (Caixas Mercado - Sem Buffer de Segurança)")
        # Horizonte Dinâmico: Apenas hoje em diante
        today = pd.Timestamp.now().normalize()
        pivot_buy = df_final[df_final['Date'] > today].pivot_table(index='Ingredient', columns='Date', values='Boxes_Buy', aggfunc='sum').fillna(0)
        pivot_buy.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_buy.columns]
        st.dataframe(pivot_buy[pivot_buy.sum(axis=1) > 0].style.format("{:.0f}"), use_container_width=True)