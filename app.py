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
st.set_page_config(page_title="PCP Verdureira - Estabilidade de Dados", layout="wide")

# ==============================================================================
# 1. FUNÇÕES DE LIMPEZA E CARGA (BLINDAGEM CONTRA "TOTAL" E DATAS INVÁLIDAS)
# ==============================================================================

def safety_clean_dataframe(df, date_col):
    """
    Remove linhas de 'Total', converte datas com segurança e limpa strings.
    """
    # 1. Remove espaços em branco dos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # 2. Converte a coluna de data. O que não for data (ex: "Total") vira NaT (nulo)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # 3. Remove linhas onde a data ficou nula (remove rodapés de 'Total')
    df = df.dropna(subset=[date_col])
    
    # 4. Filtro de segurança para anos plausíveis (evita o erro 5025)
    mask = (df[date_col].dt.year > 2020) & (df[date_col].dt.year < 2100)
    return df[mask].copy()

@st.cache_data
def load_data(uploaded_file):
    try:
        # Lê o arquivo
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # Mapeamento de colunas dinâmico
        rename_map = {
            'Data':'Date', 'Dia':'Date', 
            'Cod- SKU':'SKU', 'Código':'SKU', 
            'Produto.DS_PRODUTO':'Description', 'Descrição':'Description', 
            'Pedidos':'Orders', 'Qtde':'Orders'
        }
        df = df.rename(columns=rename_map)
        
        # LIMPEZA CRÍTICA: Remove linhas de "Total" e erros de data
        df = safety_clean_dataframe(df, 'Date')
        
        # Garante que pedidos sejam números
        df['Orders'] = pd.to_numeric(df['Orders'], errors='coerce').fillna(0)
        
        # Classificação de Grupos PCP
        def classify_group(desc):
            txt = str(desc).lower()
            if 'americana bola' in txt: return 'Americana Bola'
            if any(x in txt for x in ['vero', 'primavera', 'roxa', 'mix', 'repolho', 'couve']): return 'Vero'
            if 'mini' in txt: return 'Minis'
            if any(x in txt for x in ['salada', 'alface', 'rúcula', 'agrião', 'escarola']): return 'Saladas'
            return 'Outros'
        
        df['Group'] = df['Description'].apply(classify_group)
        
        # Agrupa para consolidar caso haja SKUs repetidos no mesmo dia
        return df.groupby(['Date','SKU','Description','Group'])['Orders'].sum().reset_index()
    except Exception as e:
        st.error(f"Erro ao processar Vendas: {e}")
        return pd.DataFrame()

@st.cache_data
def load_yield_data_scenarios(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # LIMPEZA CRÍTICA: Remove linhas de "Total" e erros de data
        df = safety_clean_dataframe(df, 'Data')
        
        df['Produto'] = df['Produto'].astype(str).str.strip().str.lower()
        df['Fornecedor'] = df['Fornecedor'].astype(str).str.upper().str.strip()
        df['Origem'] = np.where(df['Fornecedor'] == 'VERDE PRIMA', 'VP', 'MERCADO')
        
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
    except Exception as e:
        st.error(f"Erro ao processar Rendimentos: {e}")
        return pd.DataFrame()

@st.cache_data
def load_recipe_data(uploaded_file):
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Cod': 'SKU', 'Materia Prima': 'Ingredient', 'Composição (mg)': 'Weight_g', 'Tipo': 'Type'})
    df['Weight_g'] = pd.to_numeric(df['Weight_g'], errors='coerce').fillna(0)
    return df[['SKU', 'Ingredient', 'Weight_g', 'Type']]

@st.cache_data
def load_availability_data(uploaded_file):
    # Pula 2 linhas de cabeçalho comuns na planilha de colheita
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
    df = df_raw.copy()
    last_date = df['Date'].max()
    unique_skus = df[['SKU', 'Description', 'Group']].drop_duplicates()
    
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
        if d.dayofweek == 6: temp['Orders'] = 0 # Domingo Zero
        preds.append(temp)
    return pd.concat(preds)

# ==============================================================================
# 3. INTERFACE PCP
# ==============================================================================

st.title("PCP Verdureira - Gestão de Compras e Fábrica")

c1, c2 = st.columns(2)
with c1:
    f_vendas = st.file_uploader("1. Vendas (Histórico)", type=['xlsx', 'csv'])
    f_ficha = st.file_uploader("2. Ficha Técnica (Receitas)", type=['xlsx', 'csv'])
with c2:
    f_rend = st.file_uploader("3. Rendimento (Kg/Cx)", type=['xlsx', 'csv'])
    f_avail = st.file_uploader("4. Disponibilidade VP (Caixas)", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    df_raw = load_data(f_vendas)
    df_recipe = load_recipe_data(f_ficha)
    df_yield = load_yield_data_scenarios(f_rend)
    df_avail = load_availability_data(f_avail)
    
    if not df_raw.empty and not df_yield.empty:
        scenario = st.radio("Cenário de Rendimento:", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
        
        if st.button("🚀 Gerar Planejamento de Compras"):
            # 1. Previsão de Vendas
            forecast = run_forecast(df_raw)
            
            # 2. Explosão de Materiais (Kg)
            mrp = pd.merge(forecast, df_recipe, on='SKU', how='inner')
            mrp['Total_Kg'] = (mrp['Orders'] * mrp['Weight_g']) / 1000

            # REGRA: Rigidez (Não substitui se ingrediente estiver no nome do produto)
            mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)
            
            # REGRA: Antecipação de Sábado para Sexta
            mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date'] = mrp['Date'] - timedelta(days=1)
            
            # Necessidade Consolidada por Dia
            need_daily = mrp.groupby(['Date', 'Ingredient', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
            need_daily = need_daily.rename(columns={True: 'Demanda_Rigida', False: 'Demanda_Flexivel'})
            for col in ['Demanda_Rigida', 'Demanda_Flexivel']:
                if col not in need_daily: need_daily[col] = 0

            # 3. Conversão de Disponibilidade (Caixas para Kg) - REGRA DE ESPELHAMENTO
            map_dias = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
            need_daily['DayName'] = need_daily['Date'].dt.dayofweek.map(map_dias)
            
            avail_melt = df_avail.melt(id_vars='Ingredient_Key', var_name='DayName', value_name='Boxes_VP').fillna(0)
            yield_vp = df_yield[df_yield['Origem'] == 'VP'][['Produto', scenario]].rename(columns={scenario: 'Y_VP'})
            
            avail_kg = pd.merge(avail_melt, yield_vp, left_on='Ingredient_Key', right_on='Produto', how='left')
            avail_kg['Kg_VP'] = avail_kg['Boxes_VP'] * avail_kg['Y_VP'].fillna(10.0)

            # 4. Abastecimento Prioritário VP e Substituições
            df_proc = pd.merge(need_daily, avail_kg[['Ingredient_Key', 'DayName', 'Kg_VP']], 
                               left_on=['Ingredient', 'DayName'], right_on=['Ingredient_Key', 'DayName'], how='left')
            df_proc['Kg_VP'] = df_proc['Kg_VP'].fillna(0)

            final_rows = []
            for date, g in df_proc.groupby('Date'):
                # Prioridade 1: Atender Rígido com estoque VP
                g['Used_VP_Rigid'] = np.minimum(g['Kg_VP'], g['Demanda_Rigida'])
                g['Sobra_VP'] = g['Kg_VP'] - g['Used_VP_Rigid']
                
                # Prioridade 2: Atender Flexível com a própria sobra VP
                g['Used_VP_Flex'] = np.minimum(g['Sobra_VP'], g['Demanda_Flexivel'])
                
                # Déficit que gera compra de mercado
                g['Deficit_Kg'] = (g['Demanda_Rigida'] - g['Used_VP_Rigid']) + (g['Demanda_Flexivel'] - g['Used_VP_Flex'])
                final_rows.append(g)

            df_final = pd.concat(final_rows)
            
            # 5. Conversão Final para Caixas de Mercado
            yield_mkt = df_yield[df_yield['Origem'] == 'MERCADO'][['Produto', scenario]].rename(columns={scenario: 'Y_MKT'})
            df_final['Prod_Low'] = df_final['Ingredient'].str.lower().strip()
            df_final = pd.merge(df_final, yield_mkt, left_on='Prod_Low', right_on='Produto', how='left')
            df_final['Boxes_Buy'] = np.ceil(df_final['Deficit_Kg'] / df_final['Y_MKT'].fillna(10.0))

            # Exibição da Ordem de Compra
            st.subheader(f"🛒 Sugestão de Compras Diária (Caixas Mercado - Cenário {scenario})")
            # Horizonte D+1
            today = pd.Timestamp.now().normalize()
            df_view = df_final[df_final['Date'] > today].copy()
            
            pivot_buy = df_view.pivot_table(index='Ingredient', columns='Date', values='Boxes_Buy', aggfunc='sum').fillna(0)
            pivot_buy.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_buy.columns]
            
            st.dataframe(pivot_buy[pivot_buy.sum(axis=1) > 0].style.format("{:.0f}"), use_container_width=True)
            st.success("Cálculos concluídos! As linhas de 'Total' das suas planilhas foram ignoradas automaticamente.")