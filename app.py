import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import requests
import holidays
import re
import traceback
from sklearn.metrics import mean_absolute_percentage_error

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="PCP Verdureira - Industrial Intelligence v7.1", layout="wide")

# ==============================================================================
# 1. MOTOR DE INTELIGÊNCIA (CLIMA, FERIADOS E PAGAMENTO)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_smart_calendar(start_date, end_date):
    br_holidays = holidays.Brazil(subdiv='SP', state='SP')
    df_cal = pd.DataFrame({'Date': pd.date_range(start_date, end_date)})
    df_cal['IsHoliday'] = df_cal['Date'].apply(lambda x: 1 if x in br_holidays else 0).astype(int)
    df_cal['Holiday_Eve'] = df_cal['IsHoliday'].shift(-1).fillna(0).astype(int)
    df_cal['Holiday_Eve_2'] = df_cal['IsHoliday'].shift(-2).fillna(0).astype(int)
    df_cal['Is_Payday_Week'] = df_cal['Date'].dt.day.between(5, 12).astype(int)
    
    def check_special(d):
        if (d.month == 5 or d.month == 8) and d.weekday() == 6:
            if 7 < d.day <= 14: return 1
        return 0
    df_cal['IsSpecialEvent'] = df_cal['Date'].apply(check_special).astype(int)
    
    s1 = df_cal['IsSpecialEvent'].shift(-1).fillna(0)
    s2 = df_cal['IsSpecialEvent'].shift(-2).fillna(0)
    df_cal['Special_Eve'] = np.maximum(s1, s2).astype(int)

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": -23.55, "longitude": -46.63, "daily": ["temperature_2m_max"],
                  "timezone": "America/Sao_Paulo", "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d')}
        r = requests.get(url, params=params, timeout=5).json()
        df_w = pd.DataFrame({'Date': pd.to_datetime(r['daily']['time']), 'Temp_Max': r['daily']['temperature_2m_max']})
        df_cal = pd.merge(df_cal, df_w, on='Date', how='left')
    except:
        df_cal['Temp_Max'] = 25.0
        
    return df_cal.fillna(0)

# ==============================================================================
# 2. CARGA E BLINDAGEM DE TIPOS (FIX: ARROW INVALID & SKU MIXED TYPES)
# ==============================================================================

def robust_load(file, is_avail=False):
    """Carrega dados forçando SKU como String para evitar erro de PyArrow"""
    if file is None: return pd.DataFrame()
    df = pd.read_excel(file, header=2 if is_avail else 0) if file.name.endswith('xlsx') else pd.read_csv(file, sep=None, engine='python')
    
    # Remove colunas duplicadas
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # TRATAMENTO CRÍTICO DE SKU: Converte para string para aceitar '100015A'
    for col_name in ['SKU', 'Cod', 'Código', 'Cod- SKU']:
        if col_name in df.columns:
            df[col_name] = df[col_name].astype(str).str.strip().str.upper()
            
    return df

@st.cache_data
def load_and_standardize_v7(f_v, f_r, f_y, f_a):
    # VENDAS
    dv = robust_load(f_v)
    rename_v = {'Data':'Date', 'Dia':'Date', 'Cod- SKU':'SKU', 'Código':'SKU', 'Pedidos':'Orders', 'Qtde':'Orders', 'Produto.DS_PRODUTO':'Description', 'Descrição':'Description'}
    dv = dv.rename(columns=rename_v)
    dv['Date'] = pd.to_datetime(dv['Date'], errors='coerce')
    dv = dv.dropna(subset=['Date', 'SKU'])
    dv['Orders'] = pd.to_numeric(dv['Orders'], errors='coerce').fillna(0)

    # FICHA TÉCNICA
    dr = robust_load(f_r)
    if 'Cod' in dr.columns: dr = dr.rename(columns={'Cod': 'SKU'})
    dr = dr.rename(columns={'Materia Prima': 'Ingredient', 'Composição (mg)': 'Weight_g'})
    # Mantém apenas colunas únicas e essenciais
    dr = dr.loc[:, ~dr.columns.duplicated()]
    needed = [c for c in ['SKU', 'Ingredient', 'Weight_g'] if c in dr.columns]
    dr = dr[needed].copy()

    # RENDIMENTO
    dy = robust_load(f_y)
    dy['Data'] = pd.to_datetime(dy['Data'], errors='coerce')
    dy['Rendimento'] = pd.to_numeric(dy['Rendimento'], errors='coerce').fillna(10.0)

    # DISPONIBILIDADE
    da = robust_load(f_a, is_avail=True)
    return dv, dr, dy, da

# ==============================================================================
# 3. MOTOR DE APRENDIZADO (XGBOOST)
# ==============================================================================

def run_ml_forecast_v7(dv):
    df = dv.copy()
    def classify_group(desc):
        txt = str(desc).lower()
        if 'americana' in txt: return 'Americana Bola'
        if any(x in txt for x in ['vero', 'primavera', 'roxa', 'mix']): return 'Vero'
        if 'mini' in txt: return 'Minis'
        if any(x in txt for x in ['cenoura', 'beterraba', 'abobrinha', 'batata', 'legume']): return 'Legumes'
        return 'Saladas'
    df['Group'] = df['Description'].apply(classify_group)
    
    # NOVO NORMAL VERO (Jan/2025+)
    mask_vero = (df['Group'] == 'Vero') & (df['Date'] >= '2025-01-01')
    mask_others = (df['Group'] != 'Vero')
    df_train_full = df[mask_vero | mask_others].copy()
    
    last_date = df['Date'].max()
    df_cal = get_smart_calendar(df_train_full['Date'].min(), last_date + timedelta(days=7))
    df_train_full = pd.merge(df_train_full, df_cal, on='Date', how='left')
    
    df_train_full['DayOfWeek'] = df_train_full['Date'].dt.dayofweek
    df_train_full['lag_7'] = df_train_full.groupby('SKU')['Orders'].shift(7)
    df_train_full['lag_14'] = df_train_full.groupby('SKU')['Orders'].shift(14)
    
    features = ['DayOfWeek', 'lag_7', 'lag_14', 'IsHoliday', 'Holiday_Eve', 'Is_Payday_Week', 'Temp_Max']
    
    # Treino para Acurácia
    limit_date = last_date - timedelta(days=7)
    train_data = df_train_full[df_train_full['Date'] <= limit_date].dropna(subset=['lag_7', 'lag_14'])
    val_data = df_train_full[df_train_full['Date'] > limit_date].dropna(subset=['lag_7', 'lag_14'])
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
    model.fit(train_data[features], train_data['Orders'])
    
    acc = 0
    if not val_data.empty:
        p_val = model.predict(val_data[features])
        mape = mean_absolute_percentage_error(val_data['Orders'] + 1, p_val + 1)
        acc = max(0, 100 - (mape * 100))

    # Treino Final
    model.fit(df_train_full.dropna(subset=['lag_7', 'lag_14'])[features], df_train_full.dropna(subset=['lag_7', 'lag_14'])['Orders'])
    
    future_range = pd.date_range(last_date + timedelta(days=1), last_date + timedelta(days=7))
    unique_skus = df[['SKU', 'Description', 'Group']].drop_duplicates()
    
    preds_fut = []
    for d in future_range:
        temp = unique_skus.copy()
        temp['Date'] = d
        cal_dia = df_cal[df_cal['Date'] == d]
        for f in features:
            if f in cal_dia.columns: temp[f] = cal_dia[f].values[0]
        temp['DayOfWeek'] = d.dayofweek
        for l in [7, 14]:
            l_val = df[df['Date'] == (d - timedelta(days=l))][['SKU', 'Orders']].rename(columns={'Orders': f'lag_{l}'})
            temp = pd.merge(temp, l_val, on='SKU', how='left')
        temp = temp.fillna(0)
        temp['Orders'] = np.maximum(0, np.round(model.predict(temp[features])))
        if d.dayofweek == 6 or temp['IsHoliday'].iloc[0] == 1: temp['Orders'] = 0
        preds_fut.append(temp)
        
    return pd.concat(preds_fut), df_train_full, acc

# ==============================================================================
# 4. INTERFACE E LÓGICA DE NEGÓCIO PCP
# ==============================================================================

st.title("🛡️ PCP Verdureira - Gestão Industrial v7.1")

u1, u2 = st.columns(2)
with u1:
    f_vendas = st.file_uploader("1. Histórico Vendas", type=['xlsx', 'csv'])
    f_ficha = st.file_uploader("2. Ficha Técnica", type=['xlsx', 'csv'])
with u2:
    f_rend = st.file_uploader("3. Rendimento", type=['xlsx', 'csv'])
    f_avail = st.file_uploader("4. Disponibilidade VP", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    dv, dr, dy, da = load_and_standardize_v7(f_vendas, f_ficha, f_rend, f_avail)
    scenario = st.radio("Cenário de Rendimento:", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
    
    if st.button("🚀 Gerar Planejamento e Auditoria"):
        try:
            with st.spinner("IA Processando dados e blindando formatos..."):
                # 1. FORECAST
                forecast, df_hist_full, acc_val = run_ml_forecast_v7(dv)
                
                st.divider()
                st.metric("🎯 Acurácia do Cérebro (Feedback)", f"{acc_val:.1f}%")

                # 2. RESUMO EXECUTIVO TRIENAL
                st.subheader("📊 Resumo Executivo (Alinhamento Semanal)")
                f_s = forecast['Date'].min()
                ly_s, l2y_s = f_s - timedelta(days=364), f_s - timedelta(days=728)
                
                groups = ['Americana Bola', 'Vero', 'Saladas', 'Legumes', 'Minis']
                summary_list = []
                for g in groups:
                    v_curr = forecast[forecast['Group'] == g]['Orders'].sum()
                    v_ly = df_hist_full[(df_hist_full['Date'].between(ly_s, ly_s+timedelta(days=6))) & (df_hist_full['Group'] == g)]['Orders'].sum()
                    v_l2y = df_hist_full[(df_hist_full['Date'].between(l2y_s, l2y_s+timedelta(days=6))) & (df_hist_full['Group'] == g)]['Orders'].sum()
                    summary_list.append({'Grupo': g, 'Prev 2026': int(v_curr), 'Real 2025 (LY)': int(v_ly), 'Var % (25)': f"{((v_curr/v_ly)-1)*100:+.1f}%" if v_ly > 0 else "0%", 'Real 2024 (L2Y)': int(v_l2y), 'Var % (24)': f"{((v_curr/v_l2y)-1)*100:+.1f}%" if v_l2y > 0 else "0%"})
                
                df_exec = pd.DataFrame(summary_list)
                total_row = pd.DataFrame([{'Grupo': 'TOTAL GERAL', 'Prev 2026': df_exec['Prev 2026'].sum(), 'Real 2025 (LY)': df_exec['Real 2025 (LY)'].sum(), 'Var % (25)': f"{((df_exec['Prev 2026'].sum()/df_exec['Real 2025 (LY)'].sum())-1)*100:+.1f}%" if df_exec['Real 2025 (LY)'].sum() > 0 else "0%", 'Real 2024 (L2Y)': df_exec['Real 2024 (L2Y)'].sum(), 'Var % (24)': f"{((df_exec['Prev 2026'].sum()/df_exec['Real 2024 (L2Y)'].sum())-1)*100:+.1f}%" if df_exec['Real 2024 (L2Y)'].sum() > 0 else "0%"}])
                st.table(pd.concat([df_exec, total_row], ignore_index=True))

                # 3. QUADRO DE PREVISÃO (DOWNLOAD CSV)
                st.subheader("🗓️ Previsão Detalhada de Vendas (Unidades)")
                # FIX: SKU como string pura para não quebrar o Styler
                pivot_fore = forecast.copy()
                pivot_fore['SKU'] = pivot_fore['SKU'].astype(str)
                pivot_fore = pivot_fore.pivot_table(index=['SKU', 'Description'], columns='Date', values='Orders', aggfunc='sum').fillna(0)
                
                # Formata colunas de data
                map_dias = {0:'Seg', 1:'Ter', 2:'Qua', 3:'Qui', 4:'Sex', 5:'Sáb', 6:'Dom'}
                pivot_fore.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_fore.columns]
                
                # Exibição segura (usando o DataFrame puro para evitar erros de formatação do Arrow)
                st.dataframe(pivot_fore.astype(int), use_container_width=True)
                
                csv_fore = pivot_fore.to_csv().encode('utf-8')
                st.download_button("📥 Baixar Previsão de Vendas (CSV)", csv_fore, "previsao_vendas.csv", "text/csv")

                # 4. MRP E REGRAS PCP (RIGIDEZ + SÁBADO)
                mrp = pd.merge(forecast, dr, on='SKU', how='inner')
                mrp['Total_Kg'] = (mrp['Orders'] * pd.to_numeric(mrp['Weight_g'], errors='coerce')) / 1000
                mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)
                
                # Regra Sábado -> Sexta
                mrp['Date_PCP'] = mrp['Date']
                mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date_PCP'] = mrp['Date'] - timedelta(days=1)
                
                need_daily = mrp.groupby(['Date_PCP', 'Ingredient', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
                need_daily = need_daily.rename(columns={True: 'Demand_Rigid', False: 'Demand_Flex', 'Date_PCP': 'Date'})
                for c in ['Demand_Rigid', 'Demand_Flex']:
                    if c not in need_daily: need_daily[c] = 0

                # 5. ABASTECIMENTO PRIORITÁRIO VP
                need_daily['DayName'] = need_daily['Date'].dt.dayofweek.map({0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'})
                
                # Rendimentos
                y_map = []
                for (prod, forn), g in dy.groupby(['Produto', 'Fornecedor']):
                    g = g.sort_values('Data', ascending=False)
                    v_y = g['Rendimento'].iloc[0] if '1' in scenario else (g['Rendimento'].head(3).mean() if '3' in scenario else g['Rendimento'].head(5).mean())
                    y_map.append({'Produto': str(prod).lower().strip(), 'Origem': 'VP' if 'VERDE PRIMA' in str(forn).upper() else 'MKT', 'Y_Val': v_y})
                df_y_f = pd.DataFrame(y_map)

                # Disponibilidade
                name_map = {'crespa verde': 'alface crespa', 'frizzy roxa': 'frisee roxa', 'lollo': 'lollo rossa', 'chicória': 'frisee chicória'}
                da['Ing_Key'] = da['Hortaliça'].str.lower().str.strip().replace(name_map)
                avail_melt = da.melt(id_vars='Ing_Key', var_name='DayName', value_name='Boxes_VP').fillna(0)
                y_vp = df_y_f[df_yield_f['Origem'] == 'VP' if 'df_yield_f' in locals() else df_y_f['Origem'] == 'VP'].rename(columns={'Y_Val': 'Y_VP'})
                avail_kg = pd.merge(avail_melt, y_vp, left_on='Ing_Key', right_on='Produto', how='left')
                avail_kg['Kg_VP'] = pd.to_numeric(avail_kg['Boxes_VP'], errors='coerce').fillna(0) * avail_kg['Y_VP'].fillna(10.0)

                df_proc = pd.merge(need_daily, avail_kg[['Ing_Key', 'DayName', 'Kg_VP']], left_on=['Ingredient', 'DayName'], right_on=['Ing_Key', 'DayName'], how='left').fillna(0)
                
                # 6. SUBSTITUIÇÃO E DÉFICIT LÍQUIDO (FIX LOG)
                sub_log = []
                groups_sub = {'Verdes': ['alface crespa', 'escarola', 'frisee chicória', 'lalique', 'romana'], 'Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa']}
                
                final_rows = []
                for date, g in df_proc.groupby('Date'):
                    g['Used_VP_Rigid'] = np.minimum(g['Kg_VP'], g['Demand_Rigid'])
                    g['Sobra_VP'] = g['Kg_VP'] - g['Used_VP_Rigid']
                    g['Used_VP_Flex'] = np.minimum(g['Sobra_VP'], g['Demand_Flex'])
                    g['Sobra_Final'] = g['Sobra_VP'] - g['Used_VP_Flex']
                    g['Def_Flex'] = g['Demand_Flex'] - g['Used_VP_Flex']
                    
                    # Padronização de nomes para substituição
                    g['Ingredient_Lower'] = g['Ingredient'].str.lower().str.strip()
                    
                    for g_name, members in groups_sub.items():
                        mask_m = g['Ingredient_Lower'].isin(members)
                        if mask_m.any():
                            s_total = g.loc[mask_m, 'Sobra_Final'].sum()
                            f_total = g.loc[mask_m, 'Def_Flex'].sum()
                            if s_total > 0.1 and f_total > 0.1:
                                comp = min(s_total, f_total)
                                sub_log.append({'Data': date.strftime('%d/%m'), 'Grupo': g_name, 'Kg_Subst': round(comp, 1)})
                                ratio = (f_total - comp) / f_total if f_total > 0 else 0
                                g.loc[mask_m, 'Def_Flex'] *= ratio
                    
                    g['Deficit_Total'] = (g['Demand_Rigid'] - g['Used_VP_Rigid']) + g['Def_Flex']
                    final_rows.append(g)

                df_final = pd.concat(final_rows)
                y_mkt = df_y_f[df_y_f['Origem'] == 'MKT'].groupby('Produto')['Y_Val'].mean().reset_index().rename(columns={'Y_Val': 'Y_MKT'})
                df_final['Prod_Low'] = df_final['Ingredient'].str.lower().str.strip()
                df_final = pd.merge(df_final, y_mkt, left_on='Prod_Low', right_on='Produto', how='left')
                df_final['Boxes_Buy'] = np.ceil(df_final['Deficit_Total'] / df_final['Y_MKT'].fillna(10.0))

                # --- 7. EXIBIÇÃO DE RESULTADOS FINAIS ---
                st.divider()
                st.subheader("🛒 Ordem de Compra de Mercado (Caixas - D+1 em Diante)")
                today_ts = pd.Timestamp.now().normalize()
                pivot_buy = df_final[df_final['Date'] > today_ts].pivot_table(index='Ingredient', columns='Date', values='Boxes_Buy', aggfunc='sum').fillna(0)
                pivot_buy.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_buy.columns]
                st.dataframe(pivot_buy[pivot_buy.sum(axis=1) > 0].astype(int), use_container_width=True)

                c_rel1, c_rel2 = st.columns(2)
                with c_rel1:
                    st.subheader("🚜 Sobras Verde Prima (Kg)")
                    pivot_sobra = df_final[df_final['Date'] > today_ts].pivot_table(index='Ingredient', columns='Date', values='Sobra_Final', aggfunc='sum').fillna(0)
                    pivot_sobra.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_sobra.columns]
                    st.dataframe(pivot_sobra[pivot_sobra.sum(axis=1) > 0.5], use_container_width=True)
                
                with c_rel2:
                    st.subheader("🔄 Log de Substituições (Kg)")
                    if sub_log: st.table(pd.DataFrame(sub_log))
                    else: st.info("Sem substituições realizadas.")

        except Exception as e:
            st.error(f"Erro no processamento: {e}")
            st.code(traceback.format_exc())