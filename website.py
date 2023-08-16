import pandas as pd
import streamlit as st
from qfin.simulations import GeometricBrownianMotion

with st.echo(code_location='below'):

    chart_data = pd.DataFrame()

    S = st.slider('Initial Stock: ', 0,100,50,5)
    mu = st.slider('Drift %: ', 0, 30, 0, 1)
    sigma = st.slider('Volatility %: ', 0, 30, 15, 1)
    T = st.slider('T (mos): ', 0, 24, 12, 1)
    n = st.slider('Simulations: ', 0, 100, 50, 10)

    for i in range(n):
        chart_data[str(i)] = GeometricBrownianMotion(S, mu/100, sigma/100, 30, T/12).simulated_path

    st.line_chart(chart_data)
