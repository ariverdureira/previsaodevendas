import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import traceback
import re

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="PCP Verdureira - Integridade de Dados", layout="wide")

# ==============================================================================
# 1. MOTOR DE AUDITORIA (VALIDAÇÃO ANTES DO CÁLCULO)
# ==============================================================================

def audit_dataframe(df, name, date_cols=[], numeric_cols=[], mandatory_cols=[]):
    """
    Analisa o DataFrame em busca de anomalias e retorna uma lista de erros.
    """
    errors = []
    
    # 1. Verificar Colunas Duplicadas
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        errors.append(f"❌ Colunas duplicadas no arquivo {name}: {dup_cols}. Remova as colunas repetidas no Excel.")

    # 2. Verificar Colunas Obrigatórias
    missing = [c for c in mandatory_cols if c not in df.columns]
    if missing:
        errors.append(f"❌ Colunas obrigatórias ausentes no arquivo {name}: {missing}.")
        return errors # Para aqui se não tiver colunas básicas

    # 3. Validar Datas
    for col in date_cols:
        # Tenta converter e identifica onde falha
        converted = pd.to_datetime(df[col], errors='coerce')
        out_of_bounds = df[converted.isna() & df[col].notna()][col].tolist()
        if out_of_bounds:
            errors.append(f"⚠️ Datas inválidas ou fora de limite na coluna '{col}' ({name}): {out_of_bounds[:3]}... (Ex: ano 5025 ou texto 'Total').")
        
        # Verificar datas futuras absurdas (limite ano 2100)
        future_mask = (converted.dt.year > 2100)
        if future_mask.any():
            bad_dates = df[future_mask][col].tolist()
            errors.append(f"⚠️ Datas com ano incorreto detectadas em '{col}' ({name}): {bad_dates}. Corrija para o ano correto (ex: 2025).")

    # 4. Validar Números
    for col in numeric_cols:
        if col in df.columns:
            # Identifica o que não é número
            is_numeric = pd.to_numeric(df[col], errors='coerce')
            invalid_values = df[is_numeric.isna() & df[col].notna()][col].unique().tolist()
            if invalid_values:
                errors.append(f"⚠️ Valores não-numéricos na coluna '{col}' ({name}): {invalid_values}. Verifique se há textos ou símbolos.")

    return errors

# ==============================================================================
# 2. CARGAS DE DADOS COM VALIDAÇÃO EXPLÍCITA
# ==============================================================================

@st.cache_data
def load_and_audit(f_vendas, f_ficha, f_rend, f_avail):
    all_errors = []
    
    # --- 1. VENDAS ---
    df_v = pd.read_excel(f_vendas) if f_vendas.name.endswith('xlsx') else pd.read_csv(f_vendas, sep=None, engine='python')
    df_v.columns = df_v.columns.str.strip()
    rename_v = {'Data':'Date', 'Dia':'Date', 'Cod- SKU':'SKU', 'Código':'SKU', 'Produto.DS_PRODUTO':'Description', 'Descrição':'Description', 'Pedidos':'Orders', 'Qtde':'Orders'}
    df_v = df_v.rename(columns=rename_v)
    all_errors.extend(audit_dataframe(df_v, "Vendas", date_cols=['Date'], numeric_cols=['Orders'], mandatory_cols=['Date', 'SKU', 'Orders']))

    # --- 2. FICHA TÉCNICA ---
    df_r = pd.read_excel(f_ficha) if f_ficha.name.endswith('xlsx') else pd.read_csv(f_ficha, sep=None, engine='python')
    df_r.columns = df_r.columns.str.strip()
    if 'Cod' in df_r.columns: df_r = df_r.rename(columns={'Cod': 'SKU'})
    df_r = df_r.rename(columns={'Materia Prima': 'Ingredient', 'Composição (mg)': 'Weight_g'})
    all_errors.extend(audit_dataframe(df_r, "Ficha Técnica", numeric_cols=['Weight_g'], mandatory_cols=['SKU', 'Ingredient', 'Weight_g']))

    # --- 3. RENDIMENTO ---
    df_y = pd.read_excel(f_rend) if f_rend.name.endswith('xlsx') else pd.read_csv(f_rend, sep=None, engine='python')
    df_y.columns = df_y.columns.str.strip()
    all_errors.extend(audit_dataframe(df_y, "Rendimento", date_cols=['Data'], numeric_cols=['Rendimento'], mandatory_cols=['Data', 'Produto', 'Fornecedor', 'Rendimento']))

    # --- 4. DISPONIBILIDADE VP ---
    df_a = pd.read_excel(f_avail, header=2)
    df_a.columns = df_a.columns.str.strip()
    cols_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta']
    all_errors.extend(audit_dataframe(df_a, "Disponibilidade VP", numeric_cols=cols_dias, mandatory_cols=['Hortaliça'] + [c for c in cols_dias if c in df_a.columns]))

    return all_errors, df_v, df_r, df_y, df_a

# ==============================================================================
# 3. LÓGICA DE PCP (SÓ EXECUTA SE DADOS FOREM VÁLIDOS)
# ==============================================================================

def run_forecast(df_raw):
    # ML - Horizonte D+1 em diante
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    last_date = df['Date'].max()
    unique_skus = df[['SKU', 'Description']].drop_duplicates()
    
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['lag_7'] = df.groupby('SKU')['Orders'].shift(7)
    train = df.dropna(subset=['lag_7'])
    
    model = XGBRegressor(n_estimators=100)
    model.fit(train[['DayOfWeek', 'lag_7']], train['Orders'])
    
    future_range = pd.date_range(last_date + timedelta(days=1), last_date + timedelta(days=7))
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
# 4. INTERFACE STREAMLIT
# ==============================================================================

st.title("🛡️ PCP Verdureira - Auditoria de Dados & Planejamento")

# Uploads
c1, c2 = st.columns(2)
with c1:
    f_vendas = st.file_uploader("1. Histórico de Vendas", type=['xlsx', 'csv'])
    f_ficha = st.file_uploader("2. Ficha Técnica", type=['xlsx', 'csv'])
with c2:
    f_rend = st.file_uploader("3. Histórico de Rendimento", type=['xlsx', 'csv'])
    f_avail = st.file_uploader("4. Disponibilidade Verde Prima (Caixas)", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    with st.spinner("Auditando arquivos..."):
        errors, df_v, df_r, df_y, df_a = load_and_audit(f_vendas, f_ficha, f_rend, f_avail)
    
    if errors:
        st.error("### 🛑 Erros detectados nos arquivos!")
        st.write("O sistema não pode prosseguir com dados inconsistentes. Corrija as informações abaixo no seu Excel e faça o upload novamente:")
        for err in errors:
            st.warning(err)
    else:
        st.success("✅ Todos os dados foram validados com sucesso!")
        scenario = st.radio("Selecione o Cenário de Rendimento:", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
        
        if st.button("🚀 Gerar Planejamento de Fábrica e Compras"):
            # EXECUÇÃO DO PCP
            forecast = run_forecast(df_v)
            
            # MRP e Regras de Negócio
            df_r['SKU'] = df_r['SKU'].astype(str).str.strip()
            forecast['SKU'] = forecast['SKU'].astype(str).str.strip()
            
            mrp = pd.merge(forecast, df_r, on='SKU', how='inner')
            mrp['Total_Kg'] = (mrp['Orders'] * pd.to_numeric(mrp['Weight_g'])) / 1000

            # REGRA RIGIDEZ E SÁBADO -> SEXTA
            mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)
            mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date'] = mrp['Date'] - timedelta(days=1)
            
            need_daily = mrp.groupby(['Date', 'Ingredient', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
            need_daily = need_daily.rename(columns={True: 'Demanda_Rigida', False: 'Demanda_Flexivel'})
            
            # ESPELHAMENTO E CONVERSÃO VP
            map_dias = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
            need_daily['DayName'] = need_daily['Date'].dt.dayofweek.map(map_dias)
            
            # Preparar Rendimento VP
            df_y['Data'] = pd.to_datetime(df_y['Data'])
            df_y['Origem'] = np.where(df_y['Fornecedor'].str.upper().str.strip() == 'VERDE PRIMA', 'VP', 'MERCADO')
            
            yield_summary = []
            for (prod, origem), group in df_y.groupby(['Produto', 'Origem']):
                group = group.sort_values('Data', ascending=False)
                yield_summary.append({
                    'Produto': str(prod).lower().strip(), 'Origem': origem,
                    'Reativo (1)': group['Rendimento'].iloc[0],
                    'Equilibrado (3)': group['Rendimento'].head(3).mean(),
                    'Conservador (5)': group['Rendimento'].head(5).mean()
                })
            df_y_calc = pd.DataFrame(yield_summary)

            # Join Disponibilidade
            name_map = {'crespa verde': 'alface crespa', 'frizzy roxa': 'frisee roxa', 'lollo': 'lollo rossa', 'chicória': 'frisee chicória'}
            df_a['Ing_Key'] = df_a['Hortaliça'].str.lower().str.strip().replace(name_map)
            avail_melt = df_a.melt(id_vars='Ing_Key', var_name='DayName', value_name='Boxes_VP').fillna(0)
            
            y_vp = df_y_calc[df_y_calc['Origem'] == 'VP'][['Produto', scenario]].rename(columns={scenario: 'Y_VP'})
            avail_kg = pd.merge(avail_melt, y_vp, left_on='Ing_Key', right_on='Produto', how='left')
            avail_kg['Kg_VP'] = pd.to_numeric(avail_kg['Boxes_VP']) * avail_kg['Y_VP'].fillna(10.0)

            # Abastecimento
            df_proc = pd.merge(need_daily, avail_kg[['Ing_Key', 'DayName', 'Kg_VP']], 
                               left_on=['Ingredient', 'DayName'], right_on=['Ing_Key', 'DayName'], how='left')
            df_proc['Kg_VP'] = df_proc['Kg_VP'].fillna(0)

            # Cálculo de Compra
            final_rows = []
            for date, g in df_proc.groupby('Date'):
                g['Used_VP_Rigid'] = np.minimum(g['Kg_VP'], g['Demanda_Rigida'])
                g['Sobra_VP'] = g['Kg_VP'] - g['Used_VP_Rigid']
                g['Used_VP_Flex'] = np.minimum(g['Sobra_VP'], g['Demanda_Flexivel'])
                g['Deficit_Kg'] = (g['Demanda_Rigida'] - g['Used_VP_Rigid']) + (g['Demanda_Flexivel'] - g['Used_VP_Flex'])
                final_rows.append(g)

            df_final = pd.concat(final_rows)
            
            y_mkt = df_y_calc[df_y_calc['Origem'] == 'MERCADO'][['Produto', scenario]].rename(columns={scenario: 'Y_MKT'})
            df_final['Prod_Low'] = df_final['Ingredient'].str.lower().strip()
            df_final = pd.merge(df_final, y_mkt, left_on='Prod_Low', right_on='Produto', how='left')
            df_final['Boxes_Buy'] = np.ceil(df_final['Deficit_Kg'] / df_final['Y_MKT'].fillna(10.0))

            # RESULTADO
            st.subheader("🛒 Ordem de Compra de Mercado (Caixas)")
            pivot_buy = df_final[df_final['Date'] > pd.Timestamp.now()].pivot_table(index='Ingredient', columns='Date', values='Boxes_Buy', aggfunc='sum').fillna(0)
            pivot_buy.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_buy.columns]
            st.dataframe(pivot_buy[pivot_buy.sum(axis=1) > 0].style.format("{:.0f}"), use_container_width=True)