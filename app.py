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
st.set_page_config(page_title="PCP Verdureira - Gestão de Rendimento", layout="wide")

# ==============================================================================
# 1. FUNÇÕES DE CARGA E TRATAMENTO
# ==============================================================================

def classify_group(desc):
    if not isinstance(desc, str): return 'Outros'
    txt = desc.lower()
    if 'americana bola' in txt: return 'Americana Bola'
    if any(x in txt for x in ['vero', 'primavera', 'roxa', 'mix', 'repolho', 'couve']): return 'Vero'
    if 'mini' in txt: return 'Minis'
    if any(x in txt for x in ['salada', 'alface', 'rúcula', 'agrião', 'escarola']): return 'Saladas'
    return 'Outros'

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        rename_map = {'Data':'Date', 'Dia':'Date', 'Cod- SKU':'SKU', 'Código':'SKU', 'Produto.DS_PRODUTO':'Description', 'Descrição':'Description', 'Pedidos':'Orders', 'Qtde':'Orders'}
        df = df.rename(columns=rename_map)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
        df['Group'] = df['Description'].apply(classify_group)
        return df.groupby(['Date','SKU','Description','Group'])['Orders'].sum().reset_index()
    except Exception as e:
        st.error(f"Erro ao ler vendas: {e}")
        return pd.DataFrame()

@st.cache_data
def load_recipe_data(uploaded_file):
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Cod': 'SKU', 'Materia Prima': 'Ingredient', 'Composição (mg)': 'Weight_g', 'Tipo': 'Type'})
    df['Weight_g'] = pd.to_numeric(df['Weight_g'], errors='coerce').fillna(0)
    return df[['SKU', 'Ingredient', 'Weight_g', 'Type']]

@st.cache_data
def load_yield_data_scenarios(uploaded_file):
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    df['Data'] = pd.to_datetime(df['Data'])
    df['Produto'] = df['Produto'].astype(str).str.strip().str.lower()
    df['Origem'] = np.where(df['Fornecedor'].str.upper().str.strip() == 'VERDE PRIMA', 'VP', 'MERCADO')
    
    results = []
    for (prod, origem), group in df.groupby(['Produto', 'Origem']):
        group = group.sort_values('Data', ascending=False)
        results.append({
            'Produto': prod, 'Origem': origem,
            'Reativo (1)': group['Rendimento'].iloc[0],
            'Equilibrado (3)': group['Rendimento'].head(3).mean(),
            'Conservador (5)': group['Rendimento'].head(5).mean()
        })
    return pd.DataFrame(results)

@st.cache_data
def load_availability_data(uploaded_file):
    df = pd.read_excel(uploaded_file, header=2)
    df.columns = df.columns.str.strip()
    name_map = {'crespa verde': 'alface crespa', 'frizzy roxa': 'frisee roxa', 'lollo': 'lollo rossa', 'chicória': 'frisee chicória'}
    if 'Hortaliça' in df.columns:
        df = df.dropna(subset=['Hortaliça'])
        df['Ingredient_Key'] = df['Hortaliça'].str.lower().str.strip().replace(name_map)
        cols_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta']
        return df[['Ingredient_Key'] + [c for c in cols_dias if c in df.columns]]
    return pd.DataFrame()

# ==============================================================================
# 2. MOTOR DE PREVISÃO (ML)
# ==============================================================================

def run_forecast(df_raw, days=7):
    # Lógica XGBoost simplificada para o horizonte D+1 em diante
    df = df_raw.copy()
    last_date = df['Date'].max()
    unique_skus = df[['SKU', 'Description', 'Group']].drop_duplicates()
    
    # Feature Engineering
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['lag_7'] = df.groupby('SKU')['Orders'].shift(7)
    
    train = df.dropna(subset=['lag_7'])
    model = XGBRegressor(n_estimators=100)
    model.fit(train[['DayOfWeek', 'lag_7']], train['Orders'])
    
    future_range = pd.date_range(last_date + timedelta(days=1), last_date + timedelta(days=days))
    preds = []
    for d in future_range:
        temp = unique_skus.copy()
        temp['Date'] = d
        temp['DayOfWeek'] = d.dayofweek
        lag_date = d - timedelta(days=7)
        lags = df[df['Date'] == lag_date][['SKU', 'Orders']].rename(columns={'Orders': 'lag_7'})
        temp = pd.merge(temp, lags, on='SKU', how='left').fillna(0)
        temp['Orders'] = np.maximum(0, np.round(model.predict(temp[['DayOfWeek', 'lag_7']])))
        if d.dayofweek == 6: temp['Orders'] = 0
        preds.append(temp)
    return pd.concat(preds)

# ==============================================================================
# 3. LÓGICA DE PCP E INTERFACE
# ==============================================================================

st.title("Verdureira Agroindústria - PCP Inteligente")

# Uploads em abas para organizar a tela
tab_up, tab_plan = st.tabs(["📂 Upload de Dados", "🚀 Planejamento"])

with tab_up:
    c1, c2 = st.columns(2)
    with c1:
        f_vendas = st.file_uploader("Vendas", type=['xlsx', 'csv'])
        f_ficha = st.file_uploader("Ficha Técnica", type=['xlsx', 'csv'])
    with c2:
        f_rend = st.file_uploader("Rendimento (Kg/Cx)", type=['xlsx', 'csv'])
        f_avail = st.file_uploader("Disponibilidade VP (Caixas)", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    df_raw = load_data(f_vendas)
    df_recipe = load_recipe_data(f_ficha)
    df_yield = load_yield_data_scenarios(f_rend)
    df_avail = load_availability_data(f_avail)
    
    with tab_plan:
        st.subheader("Configurações de Cenário")
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            scenario = st.radio("Escolha o perfil de rendimento para conversão Caixas ↔ Kg:", 
                                ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
        
        if st.button("Executar Planejamento Completo"):
            # 1. Forecast de Vendas
            forecast = run_forecast(df_raw)
            
            # 2. MRP - Explosão de Necessidade (Kg)
            mrp = pd.merge(forecast, df_recipe, on='SKU', how='inner')
            mrp['Total_Kg'] = (mrp['Orders'] * mrp['Weight_g']) / 1000

            # REGRA: Rigidez (SKU contém ingrediente no nome)
            mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)

            # REGRA: Sábado para Sexta
            mrp['DayNum'] = mrp['Date'].dt.dayofweek
            mrp.loc[mrp['DayNum'] == 5, 'Date'] = mrp['Date'] - timedelta(days=1)
            
            # Agrupar Necessidade por Dia/Ingrediente
            need_daily = mrp.groupby(['Date', 'Ingredient', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
            need_daily = need_daily.rename(columns={True: 'Demand_Rigid', False: 'Demand_Flex'})
            if 'Demand_Rigid' not in need_daily.columns: need_daily['Demand_Rigid'] = 0
            if 'Demand_Flex' not in need_daily.columns: need_daily['Demand_Flex'] = 0

            # 3. Conversão de Disponibilidade VP (Caixas -> Kg)
            map_dias = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta'}
            need_daily['DayName'] = need_daily['Date'].dt.dayofweek.map(map_dias)
            
            avail_melt = df_avail.melt(id_vars='Ingredient_Key', var_name='DayName', value_name='Boxes_VP').fillna(0)
            
            # Trazer Rendimento VP para converter Caixas em Kg
            yield_vp = df_yield[df_yield['Origem'] == 'VP'][['Produto', scenario]].rename(columns={scenario: 'Yield_VP_Kg_Cx'})
            avail_kg = pd.merge(avail_melt, yield_vp, left_on='Ingredient_Key', right_on='Produto', how='left')
            avail_kg['Yield_VP_Kg_Cx'] = avail_kg['Yield_VP_Kg_Cx'].fillna(10.0) # Fallback
            avail_kg['Kg_VP_Available'] = avail_kg['Boxes_VP'] * avail_kg['Yield_VP_Kg_Cx']

            # 4. Cruzamento Necessidade vs Disponibilidade VP
            df_proc = pd.merge(need_daily, avail_kg[['Ingredient_Key', 'DayName', 'Kg_VP_Available']], 
                               left_on=['Ingredient', 'DayName'], right_on=['Ingredient_Key', 'DayName'], how='left')
            df_proc['Kg_VP_Available'] = df_proc['Kg_VP_Available'].fillna(0)

            # 5. Lógica de Substituição e Priorização
            groups_sub = {
                'Verdes': ['alface crespa', 'escarola', 'frisee chicória', 'lalique', 'romana'],
                'Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa']
            }

            final_rows = []
            for day, group_day in df_proc.groupby('Date'):
                # Prioridade 1: Rígido consome VP
                group_day['Used_VP_Rigid'] = np.minimum(group_day['Kg_VP_Available'], group_day['Demand_Rigid'])
                group_day['Sobra_VP'] = group_day['Kg_VP_Available'] - group_day['Used_VP_Rigid']
                
                # Prioridade 2: Flexível consome própria Sobra VP
                group_day['Used_VP_Flex'] = np.minimum(group_day['Sobra_VP'], group_day['Demand_Flex'])
                group_day['Sobra_VP'] = group_day['Sobra_VP'] - group_day['Used_VP_Flex']
                group_day['Deficit_Flex'] = group_day['Demand_Flex'] - group_day['Used_VP_Flex']

                # Prioridade 3: Substituições de Grupo (Verdes/Vermelhas)
                for g_name, items in groups_sub.items():
                    mask = group_day['Ingredient'].str.lower().isin(items)
                    if mask.any():
                        total_sobra = group_day.loc[mask, 'Sobra_VP'].sum()
                        total_falta = group_day.loc[mask, 'Deficit_Flex'].sum()
                        if total_sobra > 0 and total_falta > 0:
                            compensa = min(total_sobra, total_falta)
                            ratio = (total_falta - compensa) / total_falta
                            group_day.loc[mask, 'Deficit_Flex'] *= ratio

                final_rows.append(group_day)

            df_final = pd.concat(final_rows)

            # 6. Conversão de Déficit para Compras Mercado (Kg -> Caixas)
            yield_mkt = df_yield[df_yield['Origem'] == 'MERCADO'][['Produto', scenario]].rename(columns={scenario: 'Yield_Mkt_Kg_Cx'})
            df_final['Ingredient_Low'] = df_final['Ingredient'].str.lower().strip()
            df_final = pd.merge(df_final, yield_mkt, left_on='Ingredient_Low', right_on='Produto', how='left')
            df_final['Yield_Mkt_Kg_Cx'] = df_final['Yield_Mkt_Kg_Cx'].fillna(10.0)

            # Cálculo de Caixas de Compra: (Déficit Rígido + Déficit Flexível) / Rendimento Mercado
            df_final['Kg_Deficit_Total'] = (df_final['Demand_Rigid'] - df_final['Used_VP_Rigid']) + df_final['Deficit_Flex']
            df_final['Boxes_Buy_Mkt'] = np.ceil(df_final['Kg_Deficit_Total'] / df_final['Yield_Mkt_Kg_Cx'])

            # 7. Visualização Final
            st.divider()
            st.subheader(f"🛒 Sugestão de Compras - Mercado (Caixas) - Perfil: {scenario}")
            
            # Pivot por Data
            buy_pivot = df_final.pivot_table(index='Ingredient', columns='Date', values='Boxes_Buy_Mkt', aggfunc='sum').fillna(0)
            # Filtro Horizonte Dinâmico: Apenas hoje em diante
            buy_pivot = buy_pivot[[c for c in buy_pivot.columns if c.date() >= pd.Timestamp.now().date()]]
            
            # Formatação colunas
            buy_pivot.columns = [f"{c.strftime('%d/%m')} ({map_dias.get(c.dayofweek, 'Sáb')})" for c in buy_pivot.columns]
            buy_pivot['TOTAL'] = buy_pivot.sum(axis=1)
            
            st.dataframe(buy_pivot[buy_pivot['TOTAL'] > 0].style.format("{:.0f}"), use_container_width=True)

            # Auditoria de Rendimento Utilizado
            with st.expander("🔍 Auditoria de Conversão (Kg/Cx)", expanded=False):
                audit = df_final[['Ingredient', 'Yield_VP_Kg_Cx', 'Yield_Mkt_Kg_Cx']].drop_duplicates().dropna()
                st.table(audit)

else:
    st.warning("⚠️ Por favor, realize o upload dos 4 arquivos para habilitar o PCP.")