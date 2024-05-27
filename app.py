import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Data Analysis Application')
st.subheader('This is a simple Data Analysis Application')



# load the data
#df = sns.load_dataset('tips')

# Load the built-in datasets (Iris, Titanic, Tips)
iris_df = sns.load_dataset("iris")
titanic_df = sns.load_dataset("titanic")
tips_df = sns.load_dataset("tips")

# Create a dictionary of dataset names and corresponding DataFrames
datasets = {
    "Iris": iris_df,
    "Titanic": titanic_df,
    "Tips": tips_df,
}

# Sidebar: Dropdown to select dataset
selected_dataset = st.sidebar.selectbox("Select a dataset", list(datasets.keys()))

# Display the selected dataset
st.write(f"Selected dataset: {selected_dataset}")
st.write(datasets[selected_dataset])

# Button to upload custom dataset
if st.button("Upload Custom Dataset"):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        custom_df = pd.read_csv(uploaded_file)
        st.write("Custom dataset preview:")
        st.write(custom_df)

df = datasets[selected_dataset]

# disply the no of rows and columsn from selected table

st.write("No of Rows",datasets[selected_dataset].shape[0])
st.write("No of Rows",datasets[selected_dataset].shape[1])

# display the column name  of selected data  with  their data type



st.write("Dsiaply columns",datasets[selected_dataset].columns)

# print the null values if those are >0
if datasets[selected_dataset].isnull().sum().sum()>0:
    st.write("Null values in dataset",datasets[selected_dataset].isnull().sum().sort_values(ascending=False))
else:
    st.write("No null values in dataset")

# display summary statistics of selected data
st.write("Summary statistics of dataset",datasets[selected_dataset].describe())

# select specific columns for X and y axis from the dataset and then also select plot type to plot the data
# create a pair plot
st.subheader('Pairplot')
hue_column = st.selectbox('select column',datasets[selected_dataset].columns)
st.pyplot(sns.pairplot(datasets[selected_dataset],hue= hue_column))

st.subheader('Heat map')
# select the columns which are nummeric and then crate a corr_matrix
numeric_cols = datasets[selected_dataset].select_dtypes(include= np.number).columns
corr_matrix = datasets[selected_dataset].corr()

st.plotly_chart(sns.heatmap(corr_matrix,annot=True))











