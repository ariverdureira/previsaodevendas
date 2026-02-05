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
st.set_page_config(page_title="PCP Verdureira - Industrial Intelligence v7.8", layout="wide")

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

def clean_excel(df):
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    if not df.empty:
        df = df[~df.apply(lambda row: row.astype(str).str.contains('Total Geral|TOTAL GERAL|Soma', case=False).any(), axis=1)]
    return df

@st.cache_data
def load_all_pcp_data(f_v, f_r, f_y, f_a):
    name_fix = {'lollo': 'lollo rossa'}
    
    dv = clean_excel(pd.read_excel(f_v) if f_v.name.endswith('xlsx') else pd.read_csv(f_v, sep=None, engine='python'))
    dv = dv.rename(columns={'Data':'Date','Dia':'Date','Cod':'SKU','Cod- SKU':'SKU','Código':'SKU','Pedidos':'Orders','Qtde':'Orders','Produto.DS_PRODUTO':'Description','Descrição do código':'Description','Descrição':'Description'})
    dv['SKU'] = dv['SKU'].astype(str).str.strip().str.upper()
    dv['Description'] = dv['Description'].astype(str).str.strip()
    dv['Date'] = pd.to_datetime(dv['Date'], errors='coerce')
    dv = dv.dropna(subset=['Date', 'SKU'])
    dv['Orders'] = pd.to_numeric(dv['Orders'], errors='coerce').fillna(0)

    dr = clean_excel(pd.read_excel(f_r))
    dr = dr.rename(columns={'Cod': 'SKU', 'Materia Prima': 'Ingredient', 'Composição (mg)': 'Comp_mg'})
    dr['SKU'] = dr['SKU'].astype(str).str.strip().str.upper()
    for col in ['Ingredient', 'A', 'B', 'C']:
        if col in dr.columns:
            dr[col] = dr[col].astype(str).str.lower().str.strip().replace(name_fix)

    dy = clean_excel(pd.read_excel(f_y))
    dy['Data'] = pd.to_datetime(dy['Data'], errors='coerce')
    dy['Produto_Low'] = dy['Produto'].astype(str).str.lower().str.strip().replace(name_fix)

    da = clean_excel(pd.read_excel(f_a, header=1))
    da['Hortaliça'] = da['Hortaliça'].astype(str).str.lower().str.strip().replace(name_fix)
    
    return dv, dr, dy, da

# ==============================================================================
# 3. MOTOR DE PREVISÃO (XGBOOST)
# ==============================================================================

def run_ml_forecast(dv):
    df = dv.copy()
    def classify(desc):
        txt = str(desc).lower()
        if 'americana' in txt: return 'Americana Bola'
        if any(x in txt for x in ['vero', 'primavera', 'roxa', 'mix']): return 'Vero'
        if 'mini' in txt: return 'Minis'
        if any(x in txt for x in ['cenoura', 'beterraba', 'abobrinha', 'batata', 'legume']): return 'Legumes'
        return 'Saladas'
    df['Group'] = df['Description'].apply(classify)
    
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
    model = XGBRegressor(n_estimators=250, learning_rate=0.04, max_depth=6)
    
    train_clean = df_train_full.dropna(subset=['lag_7', 'lag_14'])
    model.fit(train_clean[features], train_clean['Orders'])
    
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
        # REGRA: Domingo e Feriado não faturam
        if d.dayofweek == 6 or temp['IsHoliday'].iloc[0] == 1: temp['Orders'] = 0
        preds_fut.append(temp)
        
    return pd.concat(preds_fut), df_train_full, df_cal[df_cal['Date'] > last_date]

# ==============================================================================
# 4. INTERFACE E LÓGICA PCP (v7.8)
# ==============================================================================

st.title("🌱 Verdureira Agroindústria - Intelligence PCP v7.8")

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
    
    if st.button("🚀 Gerar Planejamento Completo"):
        with st.spinner("Calculando demanda e esgotando colheita própria..."):
            # 1. FORECAST E CLIMA
            forecast, df_hist, weather_fut = run_ml_forecast(dv)
            
            st.divider()
            st.subheader("🌤️ Previsão do Tempo (Próximos 7 Dias)")
            w_disp = weather_fut[['Date', 'Temp_Min', 'Temp_Max', 'Chuva_mm']].copy()
            w_disp['Date'] = w_disp['Date'].dt.strftime('%d/%m (%a)')
            w_disp.columns = ['Data', 'Min (°C)', 'Max (°C)', 'Chuva (mm)']
            st.dataframe(w_disp.set_index('Data').T, use_container_width=True)
            
            # 2. RESUMO EXECUTIVO
            st.subheader("📊 Resumo Executivo (Comparativo Semana Comercial)")
            f_s = forecast['Date'].min()
            ly_s, l2y_s = f_s - timedelta(days=364), f_s - timedelta(days=728)
            
            summary_list = []
            for g in ['Americana Bola', 'Vero', 'Saladas', 'Legumes', 'Minis']:
                v_curr = forecast[forecast['Group'] == g]['Orders'].sum()
                v_ly = df_hist[(df_hist['Date'].between(ly_s, ly_s+timedelta(days=6))) & (df_hist['Group'] == g)]['Orders'].sum()
                v_l2y = df_hist[(df_hist['Date'].between(l2y_s, l2y_s+timedelta(days=6))) & (df_hist['Group'] == g)]['Orders'].sum()
                
                summary_list.append({
                    'Grupo': g, 
                    'Prev 2026': int(v_curr), 
                    'Real 2025': int(v_ly),
                    'Var % (25)': f"{((v_curr/v_ly)-1)*100:+.1f}%" if v_ly > 0 else "0%",
                    'Real 2024': int(v_l2y),
                    'Var % (24)': f"{((v_curr/v_l2y)-1)*100:+.1f}%" if v_l2y > 0 else "0%"
                })
            
            df_exec = pd.DataFrame(summary_list)
            t_curr, t_ly, t_l2y = df_exec['Prev 2026'].sum(), df_exec['Real 2025'].sum(), df_exec['Real 2024'].sum()
            total_row = pd.DataFrame([{'Grupo': 'TOTAL GERAL', 'Prev 2026': int(t_curr), 'Real 2025': int(t_ly), 'Var % (25)': f"{((t_curr/t_ly)-1)*100:+.1f}%" if t_ly > 0 else "0%", 'Real 2024': int(t_l2y), 'Var % (24)': f"{((t_curr/t_l2y)-1)*100:+.1f}%" if t_l2y > 0 else "0%"}])
            st.table(pd.concat([df_exec, total_row], ignore_index=True))

            # 3. PREVISÃO SKU NA TELA
            st.subheader("🗓️ Detalhamento da Previsão de Vendas (Unidades)")
            pivot_fore = forecast.pivot_table(index=['SKU', 'Description'], columns='Date', values='Orders', aggfunc='sum').fillna(0)
            map_dias = {0:'Seg', 1:'Ter', 2:'Qua', 3:'Qui', 4:'Sex', 5:'Sáb', 6:'Dom'}
            pivot_fore.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_fore.columns]
            st.dataframe(pivot_fore.astype(int), use_container_width=True)
            st.download_button("📥 Exportar Previsão de Vendas (CSV)", pivot_fore.to_csv().encode('utf-8'), "previsao_vendas.csv", "text/csv")

            # 4. MRP
            mrp = pd.merge(forecast, dr, on='SKU', how='inner')
            mrp['Total_Kg'] = (mrp['Orders'] * pd.to_numeric(mrp['Comp_mg'], errors='coerce')) / 1000
            mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)
            
            # SÁBADO -> SEXTA
            mrp['Date_PCP'] = mrp['Date']
            mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date_PCP'] = mrp['Date'] - timedelta(days=1)
            need_daily = mrp.groupby(['Date_PCP', 'Ingredient', 'Is_Rigid', 'A', 'B', 'C'])['Total_Kg'].sum().reset_index()

            # 5. DISPONIBILIDADE E RENDIMENTO
            da_clean = da.groupby('Hortaliça')[['Segunda','Terça','Quarta','Quinta','Sexta']].sum().reset_index()
            map_dias_ext = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
            
            y_map = []
            for (prod, forn), g in dy.groupby(['Produto_Low', 'Fornecedor']):
                g = g.sort_values('Data', ascending=False)
                if "1" in scenario_name: val = g['Rendimento'].iloc[0]
                elif "3" in scenario_name: val = g['Rendimento'].head(3).mean()
                else: val = g['Rendimento'].head(5).mean()
                y_map.append({'Produto': prod, 'Origem': 'VP' if 'VERDE' in str(forn).upper() else 'MKT', 'Y_Val': val})
            df_y_final = pd.DataFrame(y_map)

            # 6. CASCATA DE SUBSTITUIÇÃO (FIX: SOBRA DOMINGO)
            sub_log = []
            final_rows = []
            groups_sub = {
                'Verdes': ['alface crespa', 'escarola', 'frisee chicória', 'lalique', 'romana', 'espinafre', 'mini lisa'],
                'Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa']
            }
            
            for date, g_date in need_daily.groupby('Date_PCP'):
                day_name = map_dias_ext[date.dayofweek]
                
                # --- FIX: TRAVA DE FIM DE SEMANA ---
                # Se for Sábado ou Domingo, a disponibilidade real na VP é 0
                # Pois a necessidade do sábado já foi puxada para a sexta.
                if day_name in ['Sábado', 'Domingo']:
                    stock_map = {} # Zera estoque VP
                else:
                    if day_name not in da_clean.columns: day_name = 'Sexta' # Mirroring básico Seg-Sex
                    y_vp = df_y_final[df_y_final['Origem'] == 'VP'].rename(columns={'Y_Val': 'Y_VP'})
                    vp_stock_raw = da_clean[['Hortaliça', day_name]].copy().rename(columns={day_name: 'Boxes', 'Hortaliça': 'Hort_Low'})
                    vp_stock = pd.merge(vp_stock_raw, y_vp, left_on='Hort_Low', right_on='Produto', how='left')
                    vp_stock['Kg_Avail'] = vp_stock['Boxes'] * vp_stock['Y_VP'].fillna(10.0)
                    stock_map = vp_stock.set_index('Hort_Low')['Kg_Avail'].to_dict()

                # Processamento da Cascata (A -> B -> C -> Frutas)
                for idx, row in g_date.iterrows():
                    ing = str(row['Ingredient']).lower().strip()
                    needed = row['Total_Kg']
                    used_a = min(stock_map.get(ing, 0), needed)
                    stock_map[ing] = stock_map.get(ing, 0) - used_a
                    needed -= used_a
                    if not row['Is_Rigid']:
                        for alt in ['B', 'C']:
                            if needed > 0 and str(row[alt]).lower() != 'nan':
                                ing_alt = str(row[alt]).lower().strip()
                                used_alt = min(stock_map.get(ing_alt, 0), needed)
                                stock_map[ing_alt] = stock_map.get(ing_alt, 0) - used_alt
                                if used_alt > 0: sub_log.append({'Data': date.strftime('%d/%m'), 'Item': row['Ingredient'], 'Subst': row[alt], 'Kg': round(used_alt, 1), 'Origem': f'Receita {alt}'})
                                needed -= used_alt
                    g_date.at[idx, 'Deficit_Pos_Receita'] = needed

                for g_name, members in groups_sub.items():
                    mask = g_date['Ingredient'].str.lower().str.strip().isin(members) & (~g_date['Is_Rigid'])
                    for idx, row in g_date[mask].iterrows():
                        needed = row['Deficit_Pos_Receita']
                        if needed > 0:
                            for m in members:
                                if stock_map.get(m, 0) > 0 and needed > 0:
                                    take = min(stock_map[m], needed)
                                    stock_map[m] -= take
                                    needed -= take
                                    sub_log.append({'Data': date.strftime('%d/%m'), 'Item': row['Ingredient'], 'Subst': m, 'Kg': round(take, 1), 'Origem': 'Grupo '+g_name})
                        g_date.at[idx, 'Deficit_Final'] = max(0, needed)
                    
                g_date['Deficit_Final'] = g_date['Deficit_Final'].fillna(g_date['Deficit_Pos_Receita'])
                g_date['Sobra_Fazenda'] = g_date['Ingredient'].str.lower().str.strip().map(stock_map).fillna(0)
                final_rows.append(g_date)

            df_final = pd.concat(final_rows)
            y_mkt = df_y_final[df_y_final['Origem'] == 'MKT'].groupby('Produto')['Y_Val'].mean().reset_index().rename(columns={'Y_Val': 'Y_MKT'})
            df_final = pd.merge(df_final, y_mkt, left_on=df_final['Ingredient'].str.lower().str.strip(), right_on='Produto', how='left')
            df_final['Boxes_Buy'] = np.ceil(df_final['Deficit_Final'] / df_final['Y_MKT'].fillna(10.0))

            # --- 7. EXIBIÇÃO FINAL ---
            st.divider()
            st.subheader("🛒 Ordem de Compra de Mercado (Caixas - D+1)")
            today_ts = pd.Timestamp.now().normalize()
            pivot_buy = df_final[df_final['Date_PCP'] > today_ts].pivot_table(index='Ingredient', columns='Date_PCP', values='Boxes_Buy', aggfunc='sum').fillna(0)
            pivot_buy.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_buy.columns]
            st.dataframe(pivot_buy.astype(int), use_container_width=True)
            st.download_button("📥 Exportar Ordem de Compra (CSV)", pivot_buy.to_csv().encode('utf-8'), "ordem_compra.csv", "text/csv")

            col_rel1, col_rel2 = st.columns(2)
            with col_rel1:
                st.subheader("🚜 Sobras Verde Prima (Kg)")
                # Filtra apenas sobras reais (acima de 0.1kg)
                pivot_sobra = df_final[df_final['Date_PCP'] > today_ts].pivot_table(index='Ingredient', columns='Date_PCP', values='Sobra_Fazenda', aggfunc='sum').fillna(0)
                pivot_sobra = pivot_sobra[pivot_sobra.sum(axis=1) > 0.1]
                st.dataframe(pivot_sobra.style.format("{:.1f}"), use_container_width=True)
            
            with col_rel2:
                st.subheader("🔄 Log de Substituições")
                if sub_log:
                    st.download_button("📥 Exportar Log de Substituições (CSV)", pd.DataFrame(sub_log).to_csv().encode('utf-8'), "log_substituicoes.csv", "text/csv")
                    st.info(f"Substituições inteligentes realizadas para otimizar o campo.")
                else:
                    st.info("Nenhuma substituição necessária.")