import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

col1, col2= st.columns([1,3])
df = pd.read_csv("gt_2015.csv")
pol_list = [' ','CO',"NOX"]
typ_list = [' ',"Ambient","Turbine"]
amb_list = [' ',"Temperature", "Pressure", "Humidity"]
turb_list = [' ',"Exhaust pressure","Inlet temperature","After temperature","Compressor discharge pressure"]
typ = ' '
fig = plt.figure(figsize=(16,12))
sns.set_style('darkgrid')
col1.markdown("### Control Panel")
pol = col1.selectbox("Pollutants",pol_list)
if pol != ' ':
    typ = col1.selectbox("Choose Variables",typ_list)
if typ == "Ambient":
    amb = col1.selectbox("Ambient Parameters",amb_list)
    if pol == "CO":
        plt.ylim(0,25)
    if amb == "Temperature":
        sns.regplot(df,x = "AT",y = pol,lowess = True, scatter_kws = {"s":2})
    elif amb == "Pressure":
        sns.regplot(df,x = "AP",y = pol,lowess = True,scatter_kws = {"s":2})
    elif amb == "Humidity":
        sns.regplot(df,x = "AH",y = pol,lowess = True,scatter_kws = {"s":2})
    plt.xlabel(typ+" "+amb,fontsize=18)
    plt.ylabel(pol,fontsize=18)
elif typ == "Turbine":
    turb = col1.selectbox("Turbine Parameters",turb_list)
    if pol == "CO":
        plt.ylim(0,25)
    if turb == "Exhaust pressure":
        sns.regplot(df,x = "GTEP",y = pol,lowess = True,scatter_kws = {"s":2})
    elif turb == "Inlet temperature":
        sns.regplot(df,x = "TIT",y = pol,lowess = True,scatter_kws = {"s":2})
    elif turb == "After temperature":
        sns.regplot(df,x = "TAT",y = pol,lowess = True,scatter_kws = {"s":2})
    elif turb == "Compressor discharge pressure":
        sns.regplot(df,x = "CDP",y = pol,lowess = True,scatter_kws = {"s":2})
    plt.xlabel(typ+' '+turb,fontsize=18)
    plt.ylabel(pol,fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
col2.pyplot(fig)







