import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
iris = sns.load_dataset('iris')
fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_length',
                    color='species')
st.write("""
# Iris Dataset
How are sepal length, sepal width and petal_length correlated to each other?
""")
st.plotly_chart(fig)