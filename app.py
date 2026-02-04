import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import traceback

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="PCP Verdureira - Gestão Integrada", layout="wide")

# ==============================================================================
# 1. MOTOR DE AUDITORIA E CARGA DE DADOS
# ==============================================================================

def audit_dataframe(df, name, date_cols=[], numeric_cols=[], mandatory_cols=[]):
    errors = []
    # Verificar colunas obrigatórias
    missing = [c for c in mandatory_cols if c not in df.columns]
    if missing:
        errors.append(f"❌ Colunas obrigatórias ausentes no arquivo **{name}**: {missing}.")
        return errors

    # Validar Números (Garante que não haja texto onde deve haver quantidade)
    for col in numeric_cols:
        if col in df.columns:
            s_numeric = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = df[col].notna() & s_numeric.isna()
            invalid_entries = df[invalid_mask][col].unique().tolist()
            # Filtra nomes de colunas ou palavras de rodapé comuns
            invalid_entries = [x for x in invalid_entries if str(x).strip().lower() not in [col.lower(), 'total', 'subtotal']]
            if invalid_entries:
                errors.append(f"⚠️ **Erro de Tipo**: A coluna **'{col}'** em **{name}** deve conter apenas NÚMEROS. Encontramos: {invalid_entries}.")
    
    # Validar Datas (Foco no erro 5025)
    for col in date_cols:
        if col in df.columns:
            converted = pd.to_datetime(df[col], errors='coerce')
            future_mask = converted.dt.year > 2100
            if future_mask.any():
                bad_dates = df[future_mask][col].unique().tolist()
                errors.append(f"⚠️ **Data Inválida**: Encontramos anos fora do limite (ex: 5025) na coluna **'{col}'** de **{name}**: {bad_dates}.")
    
    return errors

@st.cache_data
def load_and_audit_all(f_vendas, f_ficha, f_rend, f_avail):
    all_errors = []
    
    # --- 1. VENDAS ---
    df_v = pd.read_excel(f_vendas) if f_vendas.name.endswith('xlsx') else pd.read_csv(f_vendas, sep=None, engine='python')
    df_v.columns = df_v.columns.str.strip()
    rename_v = {'Data':'Date','Dia':'Date','Cod- SKU':'SKU','Código':'SKU','Produto.DS_PRODUTO':'Description','Descrição':'Description','Pedidos':'Orders','Qtde':'Orders'}
    df_v = df_v.rename(columns=rename_v)
    df_v = df_v.loc[:, ~df_v.columns.duplicated()]
    all_errors.extend(audit_dataframe(df_v, "Vendas", date_cols=['Date'], numeric_cols=['Orders'], mandatory_cols=['Date','SKU','Orders']))

    # --- 2. FICHA TÉCNICA ---
    df_r = pd.read_excel(f_ficha) if f_ficha.name.endswith('xlsx') else pd.read_csv(f_ficha, sep=None, engine='python')
    df_r.columns = df_r.columns.str.strip()
    if 'Cod' in df_r.columns and 'SKU' in df_r.columns:
        df_r = df_r.rename(columns={'SKU': 'SKU_Name', 'Cod': 'SKU'})
    elif 'Cod' in df_r.columns:
        df_r = df_r.rename(columns={'Cod': 'SKU'})
    df_r = df_r.rename(columns={'Materia Prima': 'Ingredient', 'Composição (mg)': 'Weight_g'})
    df_r = df_r.loc[:, ~df_r.columns.duplicated()]
    all_errors.extend(audit_dataframe(df_r, "Ficha Técnica", numeric_cols=['Weight_g'], mandatory_cols=['SKU', 'Ingredient', 'Weight_g']))

    # --- 3. RENDIMENTO ---
    df_y = pd.read_excel(f_rend) if f_rend.name.endswith('xlsx') else pd.read_csv(f_rend, sep=None, engine='python')
    df_y.columns = df_y.columns.str.strip()
    all_errors.extend(audit_dataframe(df_y, "Rendimento", date_cols=['Data'], numeric_cols=['Rendimento'], mandatory_cols=['Data','Produto','Fornecedor','Rendimento']))

    # --- 4. DISPONIBILIDADE VP ---
    df_a = pd.read_excel(f_avail, header=2)
    df_a.columns = df_a.columns.str.strip()
    if 'Hortaliça' in df_a.columns:
        df_a = df_a[df_a['Hortaliça'].notna() & (df_a['Hortaliça'].astype(str).str.lower() != 'hortaliça')]
    cols_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta']
    all_errors.extend(audit_dataframe(df_a, "Disponibilidade VP", numeric_cols=cols_dias, mandatory_cols=['Hortaliça']))

    return all_errors, df_v, df_r, df_y, df_a

# ==============================================================================
# 2. MOTOR DE PREVISÃO (XGBOOST)
# ==============================================================================

def run_forecast(df_v):
    df = df_v.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    last_date = df['Date'].max()
    
    # Feature Engineering básica
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['lag_7'] = df.groupby('SKU')['Orders'].shift(7)
    train = df.dropna(subset=['lag_7'])
    
    model = XGBRegressor(n_estimators=100)
    model.fit(train[['DayOfWeek', 'lag_7']], train['Orders'])
    
    # Previsão D+1 a D+7 (Horizonte Dinâmico)
    forecast_dates = pd.date_range(last_date + timedelta(days=1), last_date + timedelta(days=7))
    unique_skus = df[['SKU', 'Description']].drop_duplicates()
    
    preds = []
    for d in forecast_dates:
        temp = unique_skus.copy()
        temp['Date'] = d
        temp['DayOfWeek'] = d.dayofweek
        lag_val = df[df['Date'] == (d - timedelta(days=7))][['SKU', 'Orders']].rename(columns={'Orders': 'lag_7'})
        temp = pd.merge(temp, lag_val, on='SKU', how='left').fillna(0)
        temp['Orders'] = np.maximum(0, np.round(model.predict(temp[['DayOfWeek', 'lag_7']])))
        if d.dayofweek == 6: temp['Orders'] = 0 
        preds.append(temp)
    return pd.concat(preds)

# ==============================================================================
# 3. INTERFACE E CÁLCULOS PCP
# ==============================================================================

st.title("🌱 Verdureira Agroindústria - PCP v4.0")

u1, u2 = st.columns(2)
with u1:
    f_vendas = st.file_uploader("1. Histórico Vendas", type=['xlsx', 'csv'])
    f_ficha = st.file_uploader("2. Ficha Técnica", type=['xlsx', 'csv'])
with u2:
    f_rend = st.file_uploader("3. Histórico Rendimento", type=['xlsx', 'csv'])
    f_avail = st.file_uploader("4. Disponibilidade Fazenda (Caixas)", type=['xlsx', 'csv'])

if f_vendas and f_ficha and f_rend and f_avail:
    errors, df_v, df_r, df_y, df_a = load_and_audit_all(f_vendas, f_ficha, f_rend, f_avail)
    
    if errors:
        st.error("### 🛑 Erros Críticos Detectados")
        st.write("Corrija os dados abaixo para liberar o planejamento:")
        for e in errors: st.warning(e)
    else:
        st.success("✅ Auditoria Concluída: Dados Integros.")
        scenario = st.radio("Cenário de Rendimento (Kg/Caixa):", ["Reativo (1)", "Equilibrado (3)", "Conservador (5)"], index=1, horizontal=True)
        
        if st.button("🚀 Gerar Planejamento de Compras"):
            with st.spinner("Processando Inteligência de Abastecimento..."):
                # 1. Forecast de Vendas
                forecast = run_forecast(df_v)
                
                # 2. Explosão de Materiais (MRP)
                df_r['SKU'] = df_r['SKU'].astype(str).str.strip()
                forecast['SKU'] = forecast['SKU'].astype(str).str.strip()
                mrp = pd.merge(forecast, df_r, on='SKU', how='inner')
                mrp['Total_Kg'] = (mrp['Orders'] * pd.to_numeric(mrp['Weight_g'], errors='coerce')) / 1000

                # REGRA: Rigidez (Não substitui se ingrediente estiver no nome do produto)
                mrp['Is_Rigid'] = mrp.apply(lambda r: str(r['Ingredient']).lower() in str(r['Description']).lower(), axis=1)
                
                # REGRA: Antecipação de Sábado para Sexta
                mrp['Date_PCP'] = mrp['Date']
                mrp.loc[mrp['Date'].dt.dayofweek == 5, 'Date_PCP'] = mrp['Date'] - timedelta(days=1)
                
                # Consolidação Diária
                need_daily = mrp.groupby(['Date_PCP', 'Ingredient', 'Is_Rigid'])['Total_Kg'].sum().unstack(fill_value=0).reset_index()
                need_daily = need_daily.rename(columns={True: 'Demanda_Rigida', False: 'Demanda_Flexivel', 'Date_PCP': 'Date'})
                for c in ['Demanda_Rigida', 'Demanda_Flexivel']:
                    if c not in need_daily: need_daily[c] = 0

                # 3. RENDIMENTOS E ESPELHAMENTO
                map_dias = {0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
                need_daily['DayName'] = need_daily['Date'].dt.dayofweek.map(map_dias)
                
                # Cálculo de Rendimento por Fornecedor
                df_y['Data'] = pd.to_datetime(df_y['Data'], errors='coerce')
                yield_summary = []
                for (prod, forn), g in df_y.groupby(['Produto', 'Fornecedor']):
                    g = g.sort_values('Data', ascending=False)
                    val = g['Rendimento'].iloc[0] if '1' in scenario else (g['Rendimento'].head(3).mean() if '3' in scenario else g['Rendimento'].head(5).mean())
                    yield_summary.append({
                        'Produto': str(prod).lower().strip(),
                        'Origem': 'VP' if 'VERDE PRIMA' in str(forn).upper() else 'MKT',
                        'Y_Val': val
                    })
                df_yield_final = pd.DataFrame(yield_summary)

                # 4. DISPONIBILIDADE VP (Conversão Caixas para Kg)
                name_map = {'crespa verde': 'alface crespa', 'frizzy roxa': 'frisee roxa', 'lollo': 'lollo rossa', 'chicória': 'frisee chicória'}
                df_a['Ing_Key'] = df_a['Hortaliça'].str.lower().str.strip().replace(name_map)
                avail_melt = df_a.melt(id_vars='Ing_Key', var_name='DayName', value_name='Boxes_VP').fillna(0)
                
                y_vp = df_yield_final[df_yield_final['Origem'] == 'VP'].rename(columns={'Y_Val': 'Y_VP'})
                avail_kg = pd.merge(avail_melt, y_vp, left_on='Ing_Key', right_on='Produto', how='left')
                avail_kg['Kg_VP'] = pd.to_numeric(avail_kg['Boxes_VP'], errors='coerce').fillna(0) * avail_kg['Y_VP'].fillna(10.0)

                # 5. ABASTECIMENTO PRIORITÁRIO E SUBSTITUIÇÃO
                df_proc = pd.merge(need_daily, avail_kg[['Ing_Key', 'DayName', 'Kg_VP']], left_on=['Ingredient', 'DayName'], right_on=['Ing_Key', 'DayName'], how='left')
                df_proc['Kg_VP'] = df_proc['Kg_VP'].fillna(0)

                groups_sub = {
                    'Verdes': ['alface crespa', 'escarola', 'frisee chicória', 'lalique', 'romana'],
                    'Vermelhas': ['frisee roxa', 'lollo rossa', 'mini lisa roxa']
                }

                final_rows = []
                for date, g in df_proc.groupby('Date'):
                    # 1. Consome Rígido do estoque VP
                    g['Used_VP_Rigid'] = np.minimum(g['Kg_VP'], g['Demanda_Rigida'])
                    g['Sobra_VP'] = g['Kg_VP'] - g['Used_VP_Rigid']
                    
                    # 2. Consome Flexível do próprio item
                    g['Used_VP_Flex'] = np.minimum(g['Sobra_VP'], g['Demanda_Flexivel'])
                    g['Sobra_VP_Item'] = g['Sobra_VP'] - g['Used_VP_Flex']
                    g['Deficit_Flex'] = g['Demanda_Flexivel'] - g['Used_VP_Flex']
                    
                    # 3. Substituições entre Grupos (Apenas Demanda Flexível)
                    for g_name, members in groups_sub.items():
                        mask = g['Ingredient'].str.lower().str.strip().isin(members)
                        if mask.any():
                            pool_sobra = g.loc[mask, 'Sobra_VP_Item'].sum()
                            pool_falta = g.loc[mask, 'Deficit_Flex'].sum()
                            if pool_sobra > 0 and pool_falta > 0:
                                comp = min(pool_sobra, pool_falta)
                                ratio = (pool_falta - comp) / pool_falta if pool_falta > 0 else 0
                                g.loc[mask, 'Deficit_Flex'] *= ratio
                    
                    # Déficit Final (Kg)
                    g['Deficit_Total_Kg'] = (g['Demanda_Rigida'] - g['Used_VP_Rigid']) + g['Deficit_Flex']
                    final_rows.append(g)

                df_final = pd.concat(final_rows)
                
                # 6. COMPRA DE MERCADO (Kg -> Caixas)
                y_mkt = df_yield_final[df_yield_final['Origem'] == 'MKT'].groupby('Produto')['Y_Val'].mean().reset_index().rename(columns={'Y_Val': 'Y_MKT'})
                df_final['Prod_Low'] = df_final['Ingredient'].str.lower().str.strip()
                df_final = pd.merge(df_final, y_mkt, left_on='Prod_Low', right_on='Produto', how='left')
                df_final['Boxes_Buy'] = np.ceil(df_final['Deficit_Total_Kg'] / df_final['Y_MKT'].fillna(10.0))

                # --- EXIBIÇÃO EM COLUNAS DIÁRIAS ---
                st.subheader(f"🛒 Ordem de Compra (Caixas Mercado - {scenario})")
                today = pd.Timestamp.now().normalize()
                df_view = df_final[df_final['Date'] > today].copy()
                
                pivot_buy = df_view.pivot_table(index='Ingredient', columns='Date', values='Boxes_Buy', aggfunc='sum').fillna(0)
                # Formata o cabeçalho: 04/02 (Qua)
                pivot_buy.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_buy.columns]
                st.dataframe(pivot_buy[pivot_buy.sum(axis=1) > 0].style.format("{:.0f}"), use_container_width=True)
                
                with st.expander("🚜 Ver Sobras Verde Prima (Kg)"):
                    pivot_sobra = df_view.pivot_table(index='Ingredient', columns='Date', values='Sobra_VP_Item', aggfunc='sum').fillna(0)
                    pivot_sobra.columns = [f"{c.strftime('%d/%m')} ({map_dias[c.dayofweek]})" for c in pivot_sobra.columns]
                    st.dataframe(pivot_sobra[pivot_sobra.sum(axis=1) > 0.5].style.format("{:.1f}"), use_container_width=True)