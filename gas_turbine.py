import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
st.title("Gas Turbine Emission Dataset")
st.sidebar.title("Choose variables")
df = pd.read_csv("gt_2015.csv")
df.columns = ["Ambient Temperature (°C)","Ambient Pressure (mbar)","Ambient Humidity (%)",
         "Air Filter Difference Pressure (mbar)","Exhaust Pressure (mbar)","Inlet Temperature (°C)",
         "After Temperature (°C)","Compressor Discharge Pressure (mbar)","Turbine Energy Yield (MWh)","CO (mg/m3)","NOx (mg/m3)"]
purpose = st.sidebar.radio("What are you concerned about?", ["Pollution", "Operation"])
col1,col2= st.columns([3,1])
if purpose is "Pollution":
    pol_list = ["CO (mg/m3)","NOx (mg/m3)"]
    typ_list = ["Ambient","Turbine"]
    amb_list = ["Ambient Temperature (°C)","Ambient Pressure (mbar)","Ambient Humidity (%)"]
    turb_list = ["Exhaust Pressure (mbar)","Inlet Temperature (°C)",
            "After Temperature (°C)","Compressor Discharge Pressure (mbar)","Turbine Energy Yield (MWh)"]
    typ = ' '
    pol = st.sidebar.selectbox("Choose Pollutant",pol_list)
    typ = st.sidebar.selectbox("Choose Variables",typ_list)
    col1.markdown(f"#### Emission type: {pol}")
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.set_style('darkgrid')
    if typ == "Ambient":
        x = st.sidebar.selectbox("Ambient Parameters",amb_list)  
    elif typ == "Turbine":
        x = st.sidebar.selectbox("Turbine Parameters",turb_list)
    st.sidebar.markdown("Select range")
    lower = st.sidebar.slider("Min", min_value=df[x].min(), max_value=df[x].max(), value=df[x].min(), step=(df[x].max()-df[x].min())/100)
    upper = st.sidebar.slider("Max", min_value=lower, max_value=df[x].max(), value=df[x].max(), step=(df[x].max()-lower)/100)
    df_new = df[(df[x]>=lower) & (df[x]<=upper)]
    sns.regplot(df_new,x=x,y=pol,scatter_kws={"s":20},line_kws={"color": "red","linewidth":5},ax=ax)
    ax.set_xlabel(x, fontsize=24)
    ax.set_ylabel(pol, fontsize=24)
    m = linregress(df_new[x], df_new[pol]).slope
    b = linregress(df_new[x],df_new[pol]).intercept
    x_single = col2.number_input(f"Enter your {x}:")
    press = col2.button("Estimate the emssion")
    if press:
        pol_single = m*x_single + b
        col2.markdown(f"Estimated {pol} is {pol_single:.2f} mg/m3")
        ax.plot(x_single,pol_single,"orange",marker='o', markersize=20,label = "Estimation")
        ax.legend(fontsize = 24)
    col1.pyplot(fig)
    col1.markdown(f"Regression Results: Slope = {m:.2f}, Intercept = {b:.2f}")
elif purpose is "Operation":
    col1.markdown("## In progress")








