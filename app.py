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
st.set_page_config(page_title="PCP Verdureira - Inteligência Máxima v5.1", layout="wide")

# ==============================================================================
# 1. MOTOR DE INTELIGÊNCIA (CLIMA, FERIADOS NERVOSOS E PAGAMENTO)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_smart_calendar(start_date, end_date):
    """Gera contexto para a IA: Feriados (Vizinhança), Clima e Ciclo de Pagamento"""
    br_holidays = holidays.Brazil(subdiv='SP')
    df_cal = pd.DataFrame({'Date': pd.date_range(start_date, end_date)})
    
    # Feriados e Vizinhança Nervosa (Picos de Véspera e Ressaca)
    df_cal['IsHoliday'] = df_cal['Date'].apply(lambda x: 1 if x in br_holidays else 0)
    df_cal['Holiday_Eve'] = df_cal['IsHoliday'].shift(-1).fillna(0)
    df_cal['Holiday_Eve_2'] = df_cal['IsHoliday'].shift(-2).fillna(0)
    df_cal['Holiday_After'] = df_cal['IsHoliday'].shift(1).fillna(0)
    
    # Datas Especiais Móveis (Mães e Pais)
    def check_special(d):
        if (d.month == 5 or d.month == 8) and d.weekday() == 6:
            if 7 < d.day <= 14: return 1
        return 0
    df_cal['IsSpecialEvent'] = df_cal['Date'].apply(check_special)
    df_cal['Special_Eve'] = df_cal['IsSpecialEvent'].shift(-1).fillna(0) | df_cal['IsSpecialEvent'].shift(-2).fillna(0)

    # Ciclo de Pagamento (Poder de compra do consumidor)
    df_cal['Is_Payday_Week'] = df_cal['Date'].dt.day.between(5, 12).astype(int)
    
    # Clima (API Open-Meteo)
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
# 2. CARGA, DEDUPLICAÇÃO E PADRONIZAÇÃO (SEGURANÇA TOTAL)
# ==============================================================================

def load_and_clean(file, is_avail=False):
    """Lê o arquivo e remove colunas duplicadas imediatamente"""
    if file is None: return pd.DataFrame()
    df = pd.read_excel(file, header=2 if is_avail else 0) if file.name.endswith('xlsx') else pd.read_csv(file, sep=None, engine='python')
    
    # REMOVE COLUNAS DUPLICADAS (Causa do AttributeError)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_pcp_data(f_v, f_r, f_y, f_a):
    # VENDAS
    dv = load_and_clean(f_v)
    dv = dv.rename(columns={'Data':'Date', 'Dia':'Date', 'Cod- SKU':'SKU', 'Código':'SKU', 'Pedidos':'Orders', 'Qtde':'Orders', 'Produto.DS_PRODUTO':'Description', 'Descrição':'Description'})
    dv['Date'] = pd.to_datetime(dv['Date'], errors='coerce')
    dv = dv.dropna(subset=['Date'])

    # FICHA TÉCNICA
    dr = load_and_clean(f_r)
    if 'Cod' in dr.columns: dr = dr.rename(columns={'Cod': 'SKU'})
    dr = dr.rename(columns={'Materia Prima': 'Ingredient', 'Composição (mg)': 'Weight_g'})
    # Força SKU a ser string e limpa espaços
    dr['SKU'] = dr['SKU'].astype(str).str.strip()

    # RENDIMENTO
    dy = load_and_clean(f_y)
    dy['Data'] = pd.to_datetime(dy['Data'], errors='coerce')

    # DISPONIBILIDADE VP
    da = load_and_clean(f_a, is_avail=True)
    
    return dv, dr, dy, da

# ==============================================================================
# 3. MOTOR DE PREVISÃO (XGBOOST - CÉREBRO)
# ==============================================================================

def run_pcp_forecast(dv):
    df = dv.copy()
    
    def classify(desc):
        txt = str(desc).lower()
        if 'americana' in txt: return 'Americana Bola'
        if any(x in txt for x in ['vero', 'primavera', 'roxa']): return 'Vero'
        return 'Saladas'
    df['Group'] = df['Description'].apply(classify)
    
    # Novo Normal Vero (Base 2025+)
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
    
    features = ['DayOfWeek', 'lag_7', 'lag_14', 'IsHoliday', 'Holiday_Eve', 'Holiday_Eve_2', 'Holiday_After', 'Special_Eve', 'Is_Payday_Week', 'Temp_Max']
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
    train_clean = df_train.dropna(subset=['lag_7', 'lag_14'])
    model.fit(train_clean[features], train_clean['Orders'])
    
    # Predição D+1 a D+7
    future_range = pd.date_range(last_date + timedelta(days=1), last_date + timedelta(days=7))
    unique_skus = df[['SKU', 'Description', 'Group']].drop_duplicates()
    
    preds = []
    for d in future_range:
        temp = unique_skus.copy()
        temp['Date'] = d
        cal_dia = df_cal[df_cal['Date'] == d]
        for f in features:
            if f in cal_dia.columns: temp[f] = cal_dia[f].values[0]
        temp['DayOfWeek'] = d.dayofweek
        
        # Lags
        l7 = df[df['Date'] == (d - timedelta(days=7))][['SKU', 'Orders']].rename(columns={'Orders': 'lag_7'})
        l14 = df[df['Date'] == (d - timedelta(days=14))][['SKU', 'Orders']].rename(columns={'Orders': 'lag_14'})
        temp = pd.merge(temp, l7, on='SKU', how='left')
        temp = pd.merge(temp, l14, on='SKU', how='left').fillna(0)
        
        temp['Orders'] = np.maximum(0, np.round(model.predict(temp[features])))
        if d.dayofweek == 6 or temp['IsHoliday'].iloc[0] == 1: temp['Orders'] = 0
        preds.append(temp)
        
    return pd.concat(preds), df_train

# ==============================================================================
# 4. INTERFACE E LOGICA DE ABASTECIMENTO (DÉFICIT LÍQUIDO)
# ==============================================================================

st.title("🌱 Verdureira Agroindústria - PCP Inteligente v5.1")

c1, c2 = st.columns(2)
with c1:
    f_vendas = st.file_uploader("1. Vendas", type=['xlsx', 'csv'])
    f_ficha = st.file_uploader("2. Ficha Técnica", type=['xlsx', 'csv'])
with c2:
    f_rend = st.file_uploader("3. Rendimento", type=['xlsx', 'csv'])
    f_avail = st.file_uploader("4. Disponibilidade VP", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    dv, dr, dy, da = load_pcp_data(f_vendas, f_ficha, f_rend, f_avail)
    scenario = st.radio("Cenário Rendimento:", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
    
    if st.button("🚀 Gerar Planejamento de Fábrica"):
        with st.spinner("A máquina está aprendendo com clima, feriados e ciclo de pagamento..."):
            # 1. FORECAST LÍQUIDO (SEM BUFFER)
            forecast, df_hist_full = run_pcp_forecast(dv)
            
            # 2. RESUMO EXECUTIVO (SEMANA COMERCIAL)
            st.divider()
            st.subheader("📊 Resumo Executivo (Comparativo Semana Comercial)")
            f_start, f_end = forecast['Date'].min(), forecast['Date'].max()
            ly_s, l2y_s = f_start - timedelta(days=364), f_start - timedelta(days=728)
            
            res = []
            for g in ['Americana Bola', 'Vero', 'Saladas']:
                v_curr = forecast[forecast['Group'] == g]['Orders'].sum()
                v_ly = df_hist_full[(df_hist_full['Date'].between(ly_s, ly_s+timedelta(days=6))) & (df_hist_full['Group'] == g)]['Orders'].sum()
                res.append({'Grupo': g, 'Prev 2026': int(v_curr), 'Real 2025': int(v_ly), 'Var %': f"{((v_curr/v_ly)-1)*100:+.1f}%" if v_ly > 0 else "0%"})
            st.table(pd.DataFrame(res))

            # 3. MRP E REGRAS MANDATÓRIAS
            forecast['SKU'] = forecast['SKU'].astype(str).str.strip()
            mrp = pd.merge(forecast, dr, on='SKU', how='inner')
            mrp['Total_Kg'] = (mrp['Orders'] * pd.to_numeric(mrp['Weight_g'])) / 1000

            # RIGIDEZ: Nome do Ingrediente no Nome do Produto
            mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)
            
            # SÁBADO -> SEXTA
            mrp['Date_PCP'] = mrp['Date']
            mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date_PCP'] = mrp['Date'] - timedelta(days=1)
            
            need_daily = mrp.groupby(['Date_PCP', 'Ingredient', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
            need_daily = need_daily.rename(columns={True: 'Demand_Rigid', False: 'Demand_Flex', 'Date_PCP': 'Date'})

            # 4. ABASTECIMENTO PRIORITÁRIO (VP) E SUBSTITUIÇÃO
            map_dias = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
            need_daily['DayName'] = need_daily['Date'].dt.dayofweek.map(map_dias)
            
            # Rendimentos
            yield_map = []
            for (prod, forn), g in dy.groupby(['Produto', 'Fornecedor']):
                g = g.sort_values('Data', ascending=False)
                val = g['Rendimento'].iloc[0] if '1' in scenario else (g['Rendimento'].head(3).mean() if '3' in scenario else g['Rendimento'].head(5).mean())
                yield_map.append({'Produto': str(prod).lower().strip(), 'Origem': 'VP' if 'VERDE PRIMA' in str(forn).upper() else 'MKT', 'Y_Val': val})
            df_y_f = pd.DataFrame(yield_map)

            # Disponibilidade
            name_map = {'crespa verde': 'alface crespa', 'frizzy roxa': 'frisee roxa', 'lollo': 'lollo rossa', 'chicória': 'frisee chicória'}
            da['Ing_Key'] = da['Hortaliça'].str.lower().str.strip().replace(name_map)
            avail_melt = da.melt(id_vars='Ing_Key', var_name='DayName', value_name='Boxes_VP').fillna(0)
            y_vp = df_y_f[df_y_f['Origem'] == 'VP'].rename(columns={'Y_Val': 'Y_VP'})
            avail_kg = pd.merge(avail_melt, y_vp, left_on='Ing_Key', right_on='Produto', how='left')
            avail_kg['Kg_VP'] = pd.to_numeric(avail_kg['Boxes_VP'], errors='coerce').fillna(0) * avail_kg['Y_VP'].fillna(10.0)

            df_proc = pd.merge(need_daily, avail_kg[['Ing_Key', 'DayName', 'Kg_VP']], left_on=['Ingredient', 'DayName'], right_on=['Ing_Key', 'DayName'], how='left').fillna(0)
            
            groups_sub = {'Verdes': ['alface crespa', 'escarola', 'frisee chicória', 'lalique', 'romana'], 'Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa']}
            
            final_rows = []
            for date, g in df_proc.groupby('Date'):
                g['Used_VP_Rigid'] = np.minimum(g['Kg_VP'], g.get('Demand_Rigid', 0))
                g['Sobra_VP'] = g['Kg_VP'] - g['Used_VP_Rigid']
                g['Used_VP_Flex'] = np.minimum(g['Sobra_VP'], g.get('Demand_Flex', 0))
                g['Sobra_Item'] = g['Sobra_VP'] - g['Used_VP_Flex']
                g['Def_Flex'] = g.get('Demand_Flex', 0) - g['Used_VP_Flex']
                
                # Substituição
                for g_name, members in groups_sub.items():
                    mask = g['Ingredient'].str.lower().str.strip().isin(members)
                    if mask.any():
                        sobra = g.loc[mask, 'Sobra_Item'].sum()
                        falta = g.loc[mask, 'Def_Flex'].sum()
                        if sobra > 0 and falta > 0:
                            ratio = (falta - min(sobra, falta)) / falta if falta > 0 else 0
                            g.loc[mask, 'Def_Flex'] *= ratio
                
                g['Deficit_Líquido_Kg'] = (g.get('Demand_Rigid', 0) - g['Used_VP_Rigid']) + g['Def_Flex']
                final_rows.append(g)

            df_final = pd.concat(final_rows)
            y_mkt = df_y_f[df_y_f['Origem'] == 'MKT'].groupby('Produto')['Y_Val'].mean().reset_index().rename(columns={'Y_Val': 'Y_MKT'})
            df_final['Prod_Low'] = df_final['Ingredient'].str.lower().str.strip()
            df_final = pd.merge(df_final, y_mkt, left_on='Prod_Low', right_on='Produto', how='left')
            df_final['Boxes_Buy'] = np.ceil(df_final['Deficit_Líquido_Kg'] / df_final['Y_MKT'].fillna(10.0))

            # RESULTADO FINAL
            st.divider()
            st.subheader("🛒 Ordem de Compra de Mercado (D+1 em Diante)")
            pivot = df_final[df_final['Date'] > pd.Timestamp.now()].pivot_table(index='Ingredient', columns='Date', values='Boxes_Buy', aggfunc='sum').fillna(0)
            pivot.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot.columns]
            st.dataframe(pivot[pivot.sum(axis=1) > 0].style.format("{:.0f}"), use_container_width=True)