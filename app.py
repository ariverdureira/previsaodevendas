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
st.set_page_config(page_title="PCP Verdureira - Inteligência Máxima v6.0", layout="wide")

# ==============================================================================
# 1. MOTOR DE INTELIGÊNCIA (CLIMA, CALENDÁRIO NERVOSO E PAGAMENTO)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_smart_calendar(start_date, end_date):
    """Gera features avançadas para a máquina aprender comportamento de demanda"""
    br_holidays = holidays.Brazil(subdiv='SP')
    df_cal = pd.DataFrame({'Date': pd.date_range(start_date, end_date)})
    
    # Feriados e Vizinhança (Nervosismo: Véspera de picos e Ressacas)
    df_cal['IsHoliday'] = df_cal['Date'].apply(lambda x: 1 if x in br_holidays else 0)
    df_cal['Holiday_Eve'] = df_cal['IsHoliday'].shift(-1).fillna(0)
    df_cal['Holiday_Eve_2'] = df_cal['IsHoliday'].shift(-2).fillna(0)
    df_cal['Holiday_After'] = df_cal['IsHoliday'].shift(1).fillna(0)
    
    # Ciclo de Pagamento (5º ao 12º dia útil - maior giro no varejo)
    df_cal['Is_Payday_Week'] = df_cal['Date'].dt.day.between(5, 12).astype(int)
    
    # Clima Real (API Open-Meteo)
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": -23.55, "longitude": -46.63, "daily": ["temperature_2m_max", "precipitation_sum"],
                  "timezone": "America/Sao_Paulo", "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
        r = requests.get(url, params=params, timeout=5).json()
        df_w = pd.DataFrame({'Date': pd.to_datetime(r['daily']['time']), 'Temp_Max': r['daily']['temperature_2m_max'], 'Rain_mm': r['daily']['precipitation_sum']})
        df_cal = pd.merge(df_cal, df_w, on='Date', how='left')
    except:
        df_cal['Temp_Max'], df_cal['Rain_mm'] = 25.0, 0.0 # Médias neutras se API falhar
        
    return df_cal.fillna(0)

# ==============================================================================
# 2. CARGA E BLINDAGEM DE DADOS (DEDUPLICAÇÃO DE COLUNAS)
# ==============================================================================

def robust_load(file, is_avail=False):
    """Carrega arquivos removendo duplicatas de colunas que causam AttributeError"""
    if file is None: return pd.DataFrame()
    df = pd.read_excel(file, header=2 if is_avail else 0) if file.name.endswith('xlsx') else pd.read_csv(file, sep=None, engine='python')
    
    # Remove colunas duplicadas no Excel (ex: duas colunas SKU)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_all_pcp_data(f_v, f_r, f_y, f_a):
    # VENDAS
    dv = robust_load(f_v)
    dv = dv.rename(columns={'Data':'Date', 'Dia':'Date', 'Cod- SKU':'SKU', 'Código':'SKU', 'Pedidos':'Orders', 'Qtde':'Orders', 'Produto.DS_PRODUTO':'Description', 'Descrição':'Description'})
    dv['Date'] = pd.to_datetime(dv['Date'], errors='coerce')
    dv = dv.dropna(subset=['Date'])

    # FICHA TÉCNICA
    dr = robust_load(f_r)
    if 'Cod' in dr.columns: dr = dr.rename(columns={'Cod': 'SKU'})
    dr = dr.rename(columns={'Materia Prima': 'Ingredient', 'Composição (mg)': 'Weight_g'})
    dr = dr[['SKU', 'Ingredient', 'Weight_g']].copy() # Isola colunas para evitar conflitos
    dr['SKU'] = dr['SKU'].astype(str).str.strip()

    # RENDIMENTO
    dy = robust_load(f_y)
    dy['Data'] = pd.to_datetime(dy['Data'], errors='coerce')

    # DISPONIBILIDADE VP
    da = robust_load(f_a, is_avail=True)
    return dv, dr, dy, da

# ==============================================================================
# 3. MOTOR DE MACHINE LEARNING (XGBOOST - CÉREBRO)
# ==============================================================================

def run_ml_forecast(dv):
    df = dv.copy()
    
    def classify_group(desc):
        txt = str(desc).lower()
        if 'americana' in txt: return 'Americana Bola'
        if any(x in txt for x in ['vero', 'primavera', 'roxa', 'mix']): return 'Vero'
        if 'mini' in txt: return 'Minis'
        if any(x in txt for x in ['cenoura', 'beterraba', 'abobrinha', 'legume']): return 'Legumes'
        return 'Saladas'
    
    df['Group'] = df['Description'].apply(classify_group)
    
    # NOVO NORMAL VERO (Jan/2025 em diante)
    mask_vero = (df['Group'] == 'Vero') & (df['Date'] >= '2025-01-01')
    mask_others = (df['Group'] != 'Vero')
    df_train = df[mask_vero | mask_others].copy()
    
    last_date = df['Date'].max()
    df_cal = get_smart_calendar(df_train['Date'].min(), last_date + timedelta(days=7))
    df_train = pd.merge(df_train, df_cal, on='Date', how='left')
    
    df_train['DayOfWeek'] = df_train['Date'].dt.dayofweek
    df_train['lag_7'] = df_train.groupby('SKU')['Orders'].shift(7)
    df_train['lag_14'] = df_train.groupby('SKU')['Orders'].shift(14)
    
    features = ['DayOfWeek', 'lag_7', 'lag_14', 'IsHoliday', 'Holiday_Eve', 'Holiday_Eve_2', 'Holiday_After', 'Is_Payday_Week', 'Temp_Max']
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
        
        # Buscar Lags Reais
        l7 = df[df['Date'] == (d - timedelta(days=7))][['SKU', 'Orders']].rename(columns={'Orders': 'lag_7'})
        l14 = df[df['Date'] == (d - timedelta(days=14))][['SKU', 'Orders']].rename(columns={'Orders': 'lag_14'})
        temp = pd.merge(temp, l7, on='SKU', how='left')
        temp = pd.merge(temp, l14, on='SKU', how='left').fillna(0)
        
        temp['Orders'] = np.maximum(0, np.round(model.predict(temp[features])))
        if d.dayofweek == 6 or temp['IsHoliday'].iloc[0] == 1: temp['Orders'] = 0
        preds.append(temp)
        
    return pd.concat(preds), df_train

# ==============================================================================
# 4. INTERFACE E LOGICA DE ABASTECIMENTO PCP
# ==============================================================================

st.title("🌱 Verdureira Agroindústria - PCP Inteligente v6.0")

u1, u2 = st.columns(2)
with u1:
    f_vendas = st.file_uploader("1. Histórico Vendas", type=['xlsx', 'csv'])
    f_ficha = st.file_uploader("2. Ficha Técnica", type=['xlsx', 'csv'])
with u2:
    f_rend = st.file_uploader("3. Rendimento", type=['xlsx', 'csv'])
    f_avail = st.file_uploader("4. Disponibilidade VP", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    dv, dr, dy, da = load_all_pcp_data(f_vendas, f_ficha, f_rend, f_avail)
    scenario = st.radio("Selecione o Cenário de Rendimento:", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
    
    if st.button("🚀 Gerar Planejamento e Relatórios"):
        with st.spinner("IA calculando picos de feriados e comparando anos anteriores..."):
            # 1. PREVISÃO
            forecast, df_hist_full = run_ml_forecast(dv)
            
            # 2. RESUMO EXECUTIVO (2026 vs 2025 vs 2024)
            st.divider()
            st.subheader("📊 Resumo Executivo (Comparativo Semana Comercial)")
            f_s, f_e = forecast['Date'].min(), forecast['Date'].max()
            # Alinhamento comercial (364 dias = 52 semanas cravadas)
            ly_s, l2y_s = f_s - timedelta(days=364), f_s - timedelta(days=728)
            
            groups = ['Americana Bola', 'Vero', 'Saladas', 'Legumes', 'Minis']
            summary_data = []
            for g in groups:
                v_curr = forecast[forecast['Group'] == g]['Orders'].sum()
                v_ly = df_hist_full[(df_hist_full['Date'].between(ly_s, ly_s+timedelta(days=6))) & (df_hist_full['Group'] == g)]['Orders'].sum()
                v_l2y = df_hist_full[(df_hist_full['Date'].between(l2y_s, l2y_s+timedelta(days=6))) & (df_hist_full['Group'] == g)]['Orders'].sum()
                summary_data.append({
                    'Grupo': g, 
                    'Prev 2026': int(v_curr), 
                    'Real 2025 (LY)': int(v_ly), 
                    'Var % (25)': f"{((v_curr/v_ly)-1)*100:+.1f}%" if v_ly > 0 else "0%",
                    'Real 2024 (L2Y)': int(v_l2y),
                    'Var % (24)': f"{((v_curr/v_l2y)-1)*100:+.1f}%" if v_l2y > 0 else "0%"
                })
            
            df_exec = pd.DataFrame(summary_data)
            # Linha de Total Geral
            total_row = {
                'Grupo': 'TOTAL GERAL',
                'Prev 2026': df_exec['Prev 2026'].sum(),
                'Real 2025 (LY)': df_exec['Real 2025 (LY)'].sum(),
                'Var % (25)': f"{((df_exec['Prev 2026'].sum()/df_exec['Real 2025 (LY)'].sum())-1)*100:+.1f}%",
                'Real 2024 (L2Y)': df_exec['Real 2024 (L2Y)'].sum(),
                'Var % (24)': f"{((df_exec['Prev 2026'].sum()/df_exec['Real 2024 (L2Y)'].sum())-1)*100:+.1f}%"
            }
            st.table(pd.concat([df_exec, pd.DataFrame([total_row])], ignore_index=True))

            # 3. MRP E REGRAS DE NEGÓCIO
            forecast['SKU'] = forecast['SKU'].astype(str).str.strip()
            mrp = pd.merge(forecast, dr, on='SKU', how='inner')
            mrp['Total_Kg'] = (mrp['Orders'] * pd.to_numeric(mrp['Weight_g'])) / 1000

            # REGRA RIGIDEZ
            mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)
            # REGRA SÁBADO -> SEXTA
            mrp['Date_PCP'] = mrp['Date']
            mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date_PCP'] = mrp['Date'] - timedelta(days=1)
            
            need_daily = mrp.groupby(['Date_PCP', 'Ingredient', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
            need_daily = need_daily.rename(columns={True: 'Demand_Rigid', False: 'Demand_Flex', 'Date_PCP': 'Date'})

            # 4. ABASTECIMENTO PRIORITÁRIO VP
            map_dias = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
            need_daily['DayName'] = need_daily['Date'].dt.dayofweek.map(map_dias)
            
            # Rendimentos Consolidados
            y_map = []
            for (prod, forn), g in dy.groupby(['Produto', 'Fornecedor']):
                g = g.sort_values('Data', ascending=False)
                val = g['Rendimento'].iloc[0] if '1' in scenario else (g['Rendimento'].head(3).mean() if '3' in scenario else g['Rendimento'].head(5).mean())
                y_map.append({'Produto': str(prod).lower().strip(), 'Origem': 'VP' if 'VERDE PRIMA' in str(forn).upper() else 'MKT', 'Y_Val': val})
            df_yield_f = pd.DataFrame(y_map)

            # Espelhamento de Colheita
            name_map = {'crespa verde': 'alface crespa', 'frizzy roxa': 'frisee roxa', 'lollo': 'lollo rossa', 'chicória': 'frisee chicória'}
            da['Ing_Key'] = da['Hortaliça'].str.lower().str.strip().replace(name_map)
            avail_melt = da.melt(id_vars='Ing_Key', var_name='DayName', value_name='Boxes_VP').fillna(0)
            y_vp = df_yield_f[df_yield_f['Origem'] == 'VP'].rename(columns={'Y_Val': 'Y_VP'})
            avail_kg = pd.merge(avail_melt, y_vp, left_on='Ing_Key', right_on='Produto', how='left')
            avail_kg['Kg_VP'] = pd.to_numeric(avail_kg['Boxes_VP'], errors='coerce').fillna(0) * avail_kg['Y_VP'].fillna(10.0)

            df_proc = pd.merge(need_daily, avail_kg[['Ing_Key', 'DayName', 'Kg_VP']], left_on=['Ingredient', 'DayName'], right_on=['Ing_Key', 'DayName'], how='left').fillna(0)
            
            # 5. SUBSTITUIÇÃO E LOG
            sub_log = []
            groups_sub = {'Verdes': ['alface crespa', 'escarola', 'frisee chicória', 'lalique', 'romana'], 'Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa']}
            
            final_rows = []
            for date, g in df_proc.groupby('Date'):
                g['Used_VP_Rigid'] = np.minimum(g['Kg_VP'], g.get('Demand_Rigid', 0))
                g['Sobra_VP'] = g['Kg_VP'] - g['Used_VP_Rigid']
                g['Used_VP_Flex'] = np.minimum(g['Sobra_VP'], g.get('Demand_Flex', 0))
                g['Sobra_Final'] = g['Sobra_VP'] - g['Used_VP_Flex']
                g['Def_Flex'] = g.get('Demand_Flex', 0) - g['Used_VP_Flex']
                
                # Regra das Substituições
                for g_name, members in groups_sub.items():
                    mask = g['Ingredient'].str.lower().str.strip().isin(members)
                    if mask.any():
                        s_total = g.loc[mask, 'Sobra_Final'].sum()
                        f_total = g.loc[mask, 'Def_Flex'].sum()
                        if s_total > 0 and f_total > 0:
                            compensa = min(s_total, f_total)
                            sub_log.append({'Data': date.strftime('%d/%m'), 'Grupo': g_name, 'Kg_Substituído': round(compensa, 1)})
                            ratio = (f_total - compensa) / f_total if f_total > 0 else 0
                            g.loc[mask, 'Def_Flex'] *= ratio
                
                g['Deficit_Líquido_Kg'] = (g.get('Demand_Rigid', 0) - g['Used_VP_Rigid']) + g['Def_Flex']
                final_rows.append(g)

            df_final = pd.concat(final_rows)
            y_mkt = df_yield_f[df_yield_f['Origem'] == 'MKT'].groupby('Produto')['Y_Val'].mean().reset_index().rename(columns={'Y_Val': 'Y_MKT'})
            df_final['Prod_Low'] = df_final['Ingredient'].str.lower().str.strip()
            df_final = pd.merge(df_final, y_mkt, left_on='Prod_Low', right_on='Produto', how='left')
            df_final['Boxes_Buy'] = np.ceil(df_final['Deficit_Líquido_Kg'] / df_final['Y_MKT'].fillna(10.0))

            # --- RESULTADOS ---
            st.divider()
            st.subheader("🛒 Ordem de Compra de Mercado (Caixas - D+1 em Diante)")
            today_ts = pd.Timestamp.now().normalize()
            pivot_buy = df_final[df_final['Date'] > today_ts].pivot_table(index='Ingredient', columns='Date', values='Boxes_Buy', aggfunc='sum').fillna(0)
            pivot_buy.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_buy.columns]
            st.dataframe(pivot_buy[pivot_buy.sum(axis=1) > 0].style.format("{:.0f}"), use_container_width=True)

            c_rel1, c_rel2 = st.columns(2)
            with c_rel1:
                st.subheader("🚜 Sobras Verde Prima (Kg)")
                pivot_sobra = df_final[df_final['Date'] > today_ts].pivot_table(index='Ingredient', columns='Date', values='Sobra_Final', aggfunc='sum').fillna(0)
                pivot_sobra.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_sobra.columns]
                st.dataframe(pivot_sobra[pivot_sobra.sum(axis=1) > 0.5].style.format("{:.1f}"), use_container_width=True)
            
            with c_rel2:
                st.subheader("🔄 Log de Substituições (Grupo Verdes/Vermelhas)")
                if sub_log: st.table(pd.DataFrame(sub_log))
                else: st.info("Nenhuma substituição realizada.")