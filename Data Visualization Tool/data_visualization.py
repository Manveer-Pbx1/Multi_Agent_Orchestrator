import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(file):
    if file.type == "text/csv":
        return pd.read_csv(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None

# Function to plot data
def plot_data(data):
    st.sidebar.header("Plotting Options")
    plot_type = st.sidebar.selectbox("Select plot type", ["Line", "Bar", "Scatter", "Histogram", "Boxplot"])
    x_axis = st.sidebar.selectbox("Select X-axis", data.columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", data.columns)

    if plot_type == "Line":
        st.line_chart(data.set_index(x_axis)[y_axis])
    elif plot_type == "Bar":
        st.bar_chart(data.set_index(x_axis)[y_axis])
    elif plot_type == "Scatter":
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
    elif plot_type == "Histogram":
        fig, ax = plt.subplots()
        sns.histplot(data[y_axis], ax=ax)
        st.pyplot(fig)
    elif plot_type == "Boxplot":
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

# Streamlit app
st.title("Interactive Data Visualization Tool")
st.write("Upload your data file (CSV or Excel) to visualize it.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.write("Data Preview:")
        st.write(data.head())
        plot_data(data)
