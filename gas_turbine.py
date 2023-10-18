import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def welcome_page():
    st.title("Gas Turbine Emission Dataset")
    mark = st.markdown("#### This is the dataset for the gas turbine emission of Carbon Monoxide(CO) and Nitrogen Oxide (NOx) at different working conditions ")
    if st.button("Enter"):
        st.session_state.page = "main"
def main_page():
    button2 = st.sidebar.button("See Description")
    if button2:
        st.session_state.page = "welcome"
        st.experimental_rerun()
    if not button2:
        st.title("Gas Turbine Emission Dataset")
        col1,col2= st.columns([3,1])
        st.sidebar.title("Choose variables")
        sns.set_style('darkgrid')
        df = pd.read_csv("gt_2015.csv")
        df.columns = ["Ambient Temperature (°C)","Ambient Pressure (mbar)","Ambient Humidity (%)",
                 "Air Filter Difference Pressure (mbar)","Exhaust Pressure (mbar)","Inlet Temperature (°C)",
                 "After Temperature (°C)","Compressor Discharge Pressure (mbar)","Turbine Energy Yield (MWh)","CO (mg/m3)","NOx (mg/m3)"]
        purpose = st.sidebar.radio("Choose the methods", ["Single variable", "Multiple variables (for ambient condition)"])
        pol_list = ["CO (mg/m3)","NOx (mg/m3)"]
        typ_list = ["Ambient","Turbine"]
        amb_list = ["Ambient Temperature (°C)","Ambient Pressure (mbar)","Ambient Humidity (%)"]
        turb_list = ["Exhaust Pressure (mbar)","Inlet Temperature (°C)",
                "After Temperature (°C)","Compressor Discharge Pressure (mbar)","Turbine Energy Yield (MWh)"]
        if purpose == "Single variable":
            typ = ' '
            pol = st.sidebar.selectbox("Choose Pollutant",pol_list)
            typ = st.sidebar.selectbox("Choose Variables",typ_list)
            col1.markdown(f"#### Emission type: {pol}")
            fig, ax = plt.subplots(figsize=(16, 12))
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
        elif purpose == "Multiple variables (for ambient condition)":
            pol = st.sidebar.selectbox("Choose Pollutant",pol_list)
            t = "Ambient Temperature (°C)"
            p = "Ambient Pressure (mbar)"
            h = "Ambient Humidity (%)"
            st.sidebar.markdown("Select Ambient Temperature Range")
            t_lower = st.sidebar.slider("Min", min_value=df[t].min(), max_value=df[t].max(), value=df[t].min(), step=(df[t].max()-df[t].min())/100)
            t_upper = st.sidebar.slider("Max", min_value=t_lower, max_value=df[t].max(), value=df[t].max(), step=(df[t].max()-t_lower)/100)
            df_t_new = df[(df[t]>=t_lower) & (df[t]<=t_upper)]
            st.sidebar.markdown("Select Ambient Pressure Range")
            p_lower = st.sidebar.slider("Min", min_value=df[p].min(), max_value=df[p].max(), value=df[p].min(), step=(df[p].max()-df[p].min())/100)
            p_upper = st.sidebar.slider("Max", min_value=p_lower, max_value=df[p].max(), value=df[p].max(), step=(df[p].max()-p_lower)/100)
            df_p_new = df[(df[p]>=p_lower) & (df[p]<=p_upper)]
            st.sidebar.markdown("Select Ambient humidity Range")
            h_lower = st.sidebar.slider("Min", min_value=df[h].min(), max_value=df[h].max(), value=df[h].min(), step=(df[h].max()-df[h].min())/100)
            h_upper = st.sidebar.slider("Max", min_value=h_lower, max_value=df[h].max(), value=df[h].max(), step=(df[h].max()-h_lower)/100)
            df_h_new = df[(df[h]>=h_lower) & (df[h]<=h_upper)]
            f,ax = plt.subplots(2,2,figsize=(16, 12))
            sns.scatterplot(df_t_new,x = t, y=pol, ax = ax[0,0])
            sns.scatterplot(df_p_new,x = p, y=pol, ax = ax[0,1])
            sns.scatterplot(df_h_new,x = h, y=pol, ax = ax[1,0])
            ax[0,0].set_xlabel(t, fontsize=24)
            ax[0,1].set_xlabel(p, fontsize=24)
            ax[1,0].set_xlabel(h, fontsize=24)
            ax[0,0].set_ylabel(pol,fontsize=24)
            ax[0,1].set_ylabel(pol,fontsize=24)
            ax[1,0].set_ylabel(pol, fontsize=24)
            df_new = df[(df[t]>=t_lower) & (df[t]<=t_upper) & (df[p]>=p_lower) & (df[p]<=p_upper) & (df[h]>=h_lower) & (df[h]<=h_upper)]
            t_min = df_new.loc[df_new[pol] == df_new[pol].min(),t]
            p_min = df_new.loc[df_new[pol] == df_new[pol].min(),p]
            h_min = df_new.loc[df_new[pol] == df_new[pol].min(),h]
            sns.scatterplot(df_new,x=t,y=p,ax=ax[1,1])
            ax[1,1].plot(t_min,p_min,"orange",marker='o', markersize=20,label = "Minimum pollution")
            ax[1,1].legend(fontsize = 24)
            col1.pyplot(f)
            col2.markdown(f"The minimum pollution in selected range is")
            col2.markdown(f" {df_new[pol].min()} °mg/m3")
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "main":
    main_page()






