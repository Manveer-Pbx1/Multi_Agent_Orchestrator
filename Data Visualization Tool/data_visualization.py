import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file):
    if file.type == "text/csv":
        return pd.read_csv(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

def plot_data(data):
    st.sidebar.header("Plotting Options")
    plot_type = st.sidebar.selectbox("Select plot type", ["Line", "Bar", "Scatter", "Histogram", "Boxplot"])
    
    if plot_type == "Histogram":
        y_axis = st.sidebar.selectbox("Select column for histogram", data.columns)
        fig, ax = plt.subplots()
        sns.histplot(data[y_axis], ax=ax)
        plt.xlabel(y_axis)
        st.pyplot(fig)
    
    elif plot_type == "Line" or plot_type == "Bar":
        x_axis = st.sidebar.selectbox("Select X-axis", data.columns)
        y_axes = st.sidebar.multiselect("Select Y-axis (multiple possible)", data.columns)
        if y_axes:
            if plot_type == "Line":
                st.line_chart(data.set_index(x_axis)[y_axes])
            else:
                st.bar_chart(data.set_index(x_axis)[y_axes])
    
    else:  
        x_axis = st.sidebar.selectbox("Select X-axis", data.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", data.columns)
        fig, ax = plt.subplots()
        if plot_type == "Scatter":
            sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
        else:  # Boxplot
            sns.boxplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

st.title("Interactive Data Visualization Tool")
st.write("Upload your data file (CSV or Excel) to visualize it.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.write("Data Preview:")
        st.write(data.head())
        plot_data(data)
