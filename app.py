import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.title('📈 Previsão de Séries Temporais - EMPRESA')


"""
### Passo 1: Importando os dados
"""
df = st.file_uploader('Importe o histórico do seu SKU aqui. As colunas serão nomedas por data e histórico de vendas', type='csv')

st.info(
            f"""
                👆 Faça upload de um arquivo .csv primeiro.
                """
        )

if df is not None:
    data = pd.read_csv(df)
    data['y'] = data['y'].str.replace(',', '').astype(float)

    data['y'] = pd.to_numeric(data['y'])
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    #st.write(max_date)

"""
### Passo 2: Selecione o Horizonte de Previsão
"""

periods_input = st.number_input('Quantos períodos (em dias) você gostaria de prever no futuro?',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)

"""
### Passo 3: Visualizar dados de previsão
A tabela abaixo mostra os valores previstos futuros. "that" é o valor previsto para as demandas futuras.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    O gráfico mostra os valores reais (pontos pretos) e previstos (linha azul) ao longo do tempo.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)

    """
    Os próximos gráficos mostram uma tendência de alto nível de valores previstos, tendências de dia da semana e tendências anuais (se o conjunto de dados abranger vários anos). A área sombreada em azul representa os intervalos de confiança superior e inferior.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)


"""
### Passo 4: Baixe os dados da previsão
O link abaixo permite que você baixe a previsão recém-criada para o seu computador para análise e uso adicionais.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Baixar arquivo de previsão</a>)'
    st.markdown(href, unsafe_allow_html=True)