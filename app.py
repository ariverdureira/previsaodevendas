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
st.set_page_config(page_title="PCP Verdureira - Rastreabilidade 48h v8.2", layout="wide")

# ==============================================================================
# 1. MOTOR DE INTELIGÊNCIA (CLIMA COMPLETO, FERIADOS E PAGAMENTO)
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
    s1, s2 = df_cal['IsSpecialEvent'].shift(-1).fillna(0), df_cal['IsSpecialEvent'].shift(-2).fillna(0)
    df_cal['Special_Eve'] = np.maximum(s1, s2).astype(int)

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": -23.55, "longitude": -46.63, 
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"], 
            "timezone": "America/Sao_Paulo", 
            "start_date": start_date.strftime('%Y-%m-%d'), 
            "end_date": end_date.strftime('%Y-%m-%d')
        }
        r = requests.get(url, params=params, timeout=5).json()
        df_w = pd.DataFrame({
            'Date': pd.to_datetime(r['daily']['time']), 
            'Temp_Max': r['daily']['temperature_2m_max'],
            'Temp_Min': r['daily']['temperature_2m_min'],
            'Chuva_mm': r['daily']['precipitation_sum']
        })
        df_cal = pd.merge(df_cal, df_w, on='Date', how='left')
    except:
        df_cal['Temp_Max'], df_cal['Temp_Min'], df_cal['Chuva_mm'] = 25.0, 18.0, 0.0
        
    return df_cal.fillna(0)

# ==============================================================================
# 2. CARGA E TRATAMENTO DE DADOS
# ==============================================================================

def normalize_name(name):
    if pd.isna(name): return ""
    n = str(name).lower().strip()
    n = n.replace("alface ", "").replace("mini ", "").replace("verde", "").strip()
    if n in ['lollo', 'lollo roxa']: n = 'lollo rossa'
    return n

def clean_excel(df):
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    if not df.empty:
        df = df[~df.apply(lambda row: row.astype(str).str.contains('Total Geral|TOTAL GERAL|Soma', case=False).any(), axis=1)]
    return df

@st.cache_data
def load_all_pcp_data(f_v, f_r, f_y, f_a):
    dv = clean_excel(pd.read_excel(f_v))
    rename_v = {'Data':'Date','Dia':'Date','Cod':'SKU','Cod- SKU':'SKU','Código':'SKU','Pedidos':'Orders','Qtde':'Orders','Produto.DS_PRODUTO':'Description','Descrição do código':'Description','Descrição':'Description'}
    dv = dv.rename(columns=rename_v)
    dv['SKU'] = dv['SKU'].astype(str).str.strip().str.upper()
    dv['Date'] = pd.to_datetime(dv['Date'], errors='coerce')
    dv = dv.dropna(subset=['Date', 'SKU'])
    
    dr = clean_excel(pd.read_excel(f_r))
    dr = dr.rename(columns={'Cod': 'SKU', 'Materia Prima': 'Ingredient', 'Composição (mg)': 'Comp_mg'})
    dr['SKU'] = dr['SKU'].astype(str).str.strip().str.upper()
    for col in ['Ingredient', 'A', 'B', 'C']:
        if col in dr.columns:
            dr[col+'_Norm'] = dr[col].apply(normalize_name)

    dy = clean_excel(pd.read_excel(f_y))
    dy['Data'] = pd.to_datetime(dy['Data'], errors='coerce')
    dy['Produto_Norm'] = dy['Produto'].apply(normalize_name)

    da = clean_excel(pd.read_excel(f_a, header=1))
    da['Hort_Norm'] = da['Hortaliça'].apply(normalize_name)
    
    return dv, dr, dy, da

# ==============================================================================
# 3. MOTOR DE PREVISÃO (XGBOOST INTELIGENTE)
# ==============================================================================

def run_ml_forecast(dv):
    df = dv.copy()
    def classify(desc):
        txt = str(desc).lower()
        if 'americana' in txt: return 'Americana Bola'
        if any(x in txt for x in ['vero', 'primavera', 'roxa']): return 'Vero'
        if 'mini' in txt: return 'Minis'
        return 'Saladas'
    df['Group'] = df['Description'].apply(classify)
    
    mask_vero = (df['Group'] == 'Vero') & (df['Date'] >= '2025-01-01')
    mask_others = (df['Group'] != 'Vero')
    df_train = df[mask_vero | mask_others].copy()
    
    last_date = df['Date'].max()
    df_cal = get_smart_calendar(df_train['Date'].min(), last_date + timedelta(days=7))
    df_train = pd.merge(df_train, df_cal, on='Date', how='left')
    
    df_train['DayOfWeek'] = df_train['Date'].dt.dayofweek
    df_train['lag_7'] = df_train.groupby('SKU')['Orders'].shift(7)
    df_train['lag_14'] = df_train.groupby('SKU')['Orders'].shift(14)
    
    features = ['DayOfWeek', 'lag_7', 'lag_14', 'IsHoliday', 'Holiday_Eve', 'Is_Payday_Week', 'Temp_Max']
    model = XGBRegressor(n_estimators=250, learning_rate=0.04, max_depth=6)
    model.fit(df_train.dropna(subset=['lag_7', 'lag_14'])[features], df_train.dropna(subset=['lag_7', 'lag_14'])['Orders'])
    
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
        
    return pd.concat(preds_fut), df_train, df_cal[df_cal['Date'] > last_date]

# ==============================================================================
# 4. INTERFACE E LÓGICA DE ABASTECIMENTO (POOL 48H FIFO)
# ==============================================================================

st.title("🌱 Verdureira Agroindústria - PCP Inteligente v8.2")

u1, u2 = st.columns(2)
with u1:
    f_vendas = st.file_uploader("1. Vendas", type=['xlsx', 'csv'])
    f_ficha = st.file_uploader("2. Ficha Técnica", type=['xlsx', 'csv'])
with u2:
    f_rend = st.file_uploader("3. Rendimento", type=['xlsx', 'csv'])
    f_avail = st.file_uploader("4. Disponibilidade VP", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    dv, dr, dy, da = load_all_pcp_data(f_vendas, f_ficha, f_rend, f_avail)
    scenario_name = st.radio("Cenário de Rendimento:", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
    
    if st.button("🚀 Gerar Planejamento de Fábrica"):
        with st.spinner("Motor de Inteligência e Lotes 48h processando..."):
            forecast, df_hist, weather_fut = run_ml_forecast(dv)
            
            # --- 1. CLIMA ---
            st.divider()
            st.subheader("🌤️ Previsão do Tempo")
            w_disp = weather_fut[['Date', 'Temp_Min', 'Temp_Max', 'Chuva_mm']].copy()
            w_disp['Date'] = w_disp['Date'].dt.strftime('%d/%m (%a)')
            st.dataframe(w_disp.set_index('Date').T, use_container_width=True)
            
            # --- 2. RESUMO EXECUTIVO ---
            st.subheader("📊 Resumo Executivo (Comparativo Semana Comercial)")
            f_s = forecast['Date'].min()
            ly_s, l2y_s = f_s - timedelta(days=364), f_s - timedelta(days=728)
            res_list = []
            for g in ['Americana Bola', 'Vero', 'Saladas', 'Minis']:
                v_curr = forecast[forecast['Group'] == g]['Orders'].sum()
                v_ly = df_hist[(df_hist['Date'].between(ly_s, ly_s+timedelta(days=6))) & (df_hist['Group'] == g)]['Orders'].sum()
                v_l2y = df_hist[(df_hist['Date'].between(l2y_s, l2y_s+timedelta(days=6))) & (df_hist['Group'] == g)]['Orders'].sum()
                res_list.append({'Grupo': g, 'IA 2026': int(v_curr), 'Real 2025': int(v_ly), 'Var %': f"{((v_curr/v_ly)-1)*100:+.1f}%" if v_ly > 0 else "0%", 'Real 2024': int(v_l2y)})
            df_exec = pd.DataFrame(res_list)
            st.table(df_exec)

            # --- 3. PREVISÃO SKU ---
            st.subheader("🗓️ Previsão SKU/Dia (Unidades)")
            pivot_fore = forecast.pivot_table(index=['SKU', 'Description'], columns='Date', values='Orders', aggfunc='sum').fillna(0)
            st.dataframe(pivot_fore.astype(int), use_container_width=True)
            st.download_button("📥 Baixar Previsão CSV", pivot_fore.to_csv().encode('utf-8'), "previsao.csv", "text/csv")

            # --- 4. CASCATA PCP COM ROLAGEM E RASTREABILIDADE ---
            mrp = pd.merge(forecast, dr, on='SKU', how='inner')
            mrp['Total_Kg'] = (mrp['Orders'] * pd.to_numeric(mrp['Comp_mg'], errors='coerce')) / 1000
            mrp['Is_Rigid'] = mrp.apply(lambda r: normalize_name(r['Ingredient']) in normalize_name(r['Description']), axis=1)
            
            # Sábado -> Sexta
            mrp['Date_Calc'] = mrp['Date']
            mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date_Calc'] = mrp['Date'] - timedelta(days=1)
            need_daily = mrp.groupby(['Date_Calc', 'Ingredient_Norm', 'Is_Rigid', 'A_Norm', 'B_Norm', 'C_Norm', 'Ingredient'])['Total_Kg'].sum().reset_index()

            da_clean = da.groupby('Hort_Norm')[['Segunda','Terça','Quarta','Quinta','Sexta']].sum().reset_index()
            y_map = []
            for (prod, forn), g in dy.groupby(['Produto_Norm', 'Fornecedor']):
                g = g.sort_values('Data', ascending=False)
                val = g['Rendimento'].iloc[0] if "1" in scenario_name else (g['Rendimento'].head(3).mean() if "3" in scenario_name else g['Rendimento'].head(5).mean())
                y_map.append({'Produto_Norm': prod, 'Origem': 'VP' if 'VERDE' in str(forn).upper() else 'MKT', 'Y_Val': val})
            df_y_f = pd.DataFrame(y_map)

            # ESTRUTURAS DE CONTROLE
            pool_estoque = {} # {item: [{'qty', 'expiry', 'added_on'}]}
            sub_log = []
            rollover_log = [] # Para o novo relatório de 24h/48h
            final_rows = []
            map_dias_ext = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
            groups_sub = {
                'Verdes': ['crespa', 'escarola', 'chicória', 'frisee chicória', 'lalique', 'romana', 'espinafre', 'mini lisa', 'agrião', 'mini agrião'],
                'Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa']
            }

            for date, g_date in need_daily.sort_values('Date_Calc').groupby('Date_Calc'):
                day_name = map_dias_ext[date.dayofweek]
                
                # Snapshot do Rollover ANTES da nova colheita (O que sobrou de ontem)
                for item, lots in pool_estoque.items():
                    qty_24h = sum(l['qty'] for l in lots if l['expiry'] == date + timedelta(days=1))
                    qty_48h = sum(l['qty'] for l in lots if l['expiry'] == date)
                    if qty_24h > 0 or qty_48h > 0:
                        rollover_log.append({'Data': date.strftime('%d/%m'), 'Item': item, 'Sobra_24h': round(qty_24h,1), 'Sobra_48h': round(qty_48h,1)})

                # 1. Adicionar Nova Colheita (Validade 48h)
                if day_name not in ['Sábado', 'Domingo']:
                    target_col = day_name if day_name in da_clean.columns else 'Sexta'
                    y_vp = df_y_f[df_y_f['Origem'] == 'VP'].rename(columns={'Y_Val': 'Y_VP'})
                    col_raw = da_clean[['Hort_Norm', target_col]].copy().rename(columns={target_col: 'Boxes', 'Hort_Norm': 'Produto_Norm'})
                    col_kg = pd.merge(col_raw, y_vp, on='Produto_Norm', how='left')
                    col_kg['Kg'] = col_kg['Boxes'] * col_kg['Y_VP'].fillna(10.0)
                    
                    for _, row_c in col_kg.iterrows():
                        item = row_c['Produto_Norm']
                        if item not in pool_estoque: pool_estoque[item] = []
                        if row_c['Kg'] > 0:
                            pool_estoque[item].append({'qty': row_c['Kg'], 'expiry': date + timedelta(days=2)})

                # 2. Limpeza de Expirados
                for item in pool_estoque:
                    pool_estoque[item] = [l for l in pool_estoque[item] if l['expiry'] > date and l['qty'] > 0]

                # Função de Consumo FIFO
                def consume_fifo(item_name, amount):
                    if item_name not in pool_estoque: return 0
                    taken = 0
                    # Ordena por expiry para comer o mais velho primeiro
                    pool_estoque[item_name] = sorted(pool_estoque[item_name], key=lambda x: x['expiry'])
                    for lot in pool_estoque[item_name]:
                        if amount <= 0: break
                        draw = min(lot['qty'], amount)
                        lot['qty'] -= draw
                        amount -= draw
                        taken += draw
                    return taken

                # 3. Cascata de Consumo
                for idx, row in g_date.iterrows():
                    ing = row['Ingredient_Norm']
                    needed = row['Total_Kg']
                    # A
                    draw = consume_fifo(ing, needed)
                    needed -= draw
                    # B/C
                    if not row['Is_Rigid'] and needed > 0:
                        for alt in ['B_Norm', 'C_Norm']:
                            if needed > 0 and str(row[alt]) != "":
                                draw_alt = consume_fifo(row[alt], needed)
                                if draw_alt > 0: sub_log.append({'Data': date.strftime('%d/%m'), 'Item': row['Ingredient'], 'Subst': row[alt], 'Kg': round(draw_alt, 1), 'Origem': f'Receita {alt}'})
                                needed -= draw_alt
                    g_date.at[idx, 'Def_Pos_Rec'] = needed

                # 4. Grupo de Frutas
                for g_name, members in groups_sub.items():
                    mask = g_date['Ingredient_Norm'].isin(members) & (~g_date['Is_Rigid'])
                    for idx, row in g_date[mask].iterrows():
                        needed = row['Def_Pos_Rec']
                        if needed > 0:
                            for m in members:
                                if needed <= 0: break
                                draw_g = consume_fifo(m, needed)
                                if draw_g > 0:
                                    sub_log.append({'Data': date.strftime('%d/%m'), 'Item': row['Ingredient'], 'Subst': m, 'Kg': round(draw_g, 1), 'Origem': 'Grupo '+g_name})
                                    needed -= draw_g
                        g_date.at[idx, 'Def_Final'] = max(0, needed)
                
                g_date['Def_Final'] = g_date['Def_Final'].fillna(g_date['Def_Pos_Rec'])
                g_date['Sobra_Fazenda'] = g_date['Ingredient_Norm'].apply(lambda x: sum([l['qty'] for l in pool_estoque.get(x, [])]))
                
                # Zera pool no fim de semana
                if day_name == 'Domingo': pool_estoque = {}
                
                final_rows.append(g_date)

            df_final = pd.concat(final_rows)
            y_mkt = df_y_f[df_y_f['Origem'] == 'MKT'].groupby('Produto_Norm')['Y_Val'].mean().reset_index().rename(columns={'Y_Val': 'Y_MKT'})
            df_final = pd.merge(df_final, y_mkt, on='Produto_Norm', how='left')
            df_final['Boxes_Buy'] = np.ceil(df_final['Def_Final'] / df_final['Y_MKT'].fillna(10.0))

            # --- 5. EXIBIÇÃO DE RESULTADOS ---
            st.divider()
            st.subheader("🛒 Ordem de Compra de Mercado (Caixas)")
            pivot_buy = df_final[df_final['Date_Calc'] > pd.Timestamp.now()].pivot_table(index='Ingredient', columns='Date_Calc', values='Boxes_Buy', aggfunc='sum').fillna(0)
            pivot_buy.columns = [f"{c.strftime('%d/%m')} ({map_dias_ext[c.dayofweek][:3]})" for c in pivot_buy.columns]
            st.dataframe(pivot_buy.astype(int), use_container_width=True)
            st.download_button("📥 Baixar Ordem de Compra (CSV)", pivot_buy.to_csv().encode('utf-8'), "ordem_compra.csv", "text/csv")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("🚜 Sobras Finais na Fazenda (Kg)")
                pivot_sobra = df_final[df_final['Date_Calc'] > pd.Timestamp.now()].pivot_table(index='Ingredient', columns='Date_Calc', values='Sobra_Fazenda', aggfunc='sum').fillna(0)
                st.dataframe(pivot_sobra[pivot_sobra.sum(axis=1) > 0.1].style.format("{:.1f}"), use_container_width=True)
            with col_b:
                st.subheader("📋 Relatórios de Auditoria")
                if rollover_log:
                    st.download_button("📥 Exportar Relatório de Rolagem (24h/48h)", pd.DataFrame(rollover_log).to_csv().encode('utf-8'), "relatorio_rolagem.csv", "text/csv")
                    st.caption("O relatório de rolagem detalha quanto estoque de 'ontem' e 'anteontem' foi disponibilizado para o dia.")
                if sub_log:
                    st.download_button("📥 Exportar Log de Substituições (CSV)", pd.DataFrame(sub_log).to_csv().encode('utf-8'), "log_substituicoes.csv", "text/csv")