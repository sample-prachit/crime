import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
women_data = pd.read_csv('Data/women/combined_data.csv')

# Data Processing
women_data['Total'] = women_data.iloc[:, 5:].sum(axis=1)
women_data['STATE/UT'] = women_data['STATE/UT'].str.title().str.upper()
year_wise = women_data.groupby('Year')['Total'].sum()
state_wise = women_data.groupby('STATE/UT')['Total'].sum().sort_values()

# Handle small states in pie chart
state_wise_other = state_wise[state_wise/state_wise.sum() >= 0.025]
others = state_wise[state_wise/state_wise.sum() < 0.025].sum()
state_wise_other['Others'] = others

# Streamlit App
st.set_page_config(page_title='Victim Data Analysis', layout='wide')
st.title('Victim Data Analysis')
st.markdown("""
    <style>
        .reportview-container {
            background: #f9f9f9;
        }
        h1, h2 {
            color: #4a8cff;
        }
        .streamlit-expanderHeader {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.header('Navigation')
st.sidebar.selectbox('Select a section:', ['Overview', 'Year-wise Data', 'State-wise Data'])

# Overview Section
st.subheader('Overview')
st.write("This application visualizes victim data over the years and across different states/UTs. "
         "Use the sidebar to navigate through the data.")

# Plotting Year-wise Victims
st.subheader('Year-wise Victims')
plt.figure(figsize=(10, 5))
year_wise.plot(kind='line', marker='o', color='skyblue')
plt.title('Year-wise Victims')
plt.xlabel('Year')
plt.ylabel('Number of Victims')
plt.grid(True)
st.pyplot(plt)

# Plotting State-wise Victims (Bar Chart)
st.subheader('State-wise Victims (Bar Chart)')
plt.figure(figsize=(10, 10))
state_wise.plot(kind='barh', color='skyblue')
plt.title('State-wise Victims')
plt.xlabel('Number of Victims')
plt.ylabel('State/UT')
st.pyplot(plt)

# Plotting State-wise Victims (Pie Chart)
st.subheader('State-wise Victims (Pie Chart)')
plt.figure(figsize=(10, 5))
state_wise_other.plot(kind='pie', autopct='%1.1f%%', startangle=140, pctdistance=0.85)
plt.title('Victims Distribution by State/UT')
plt.ylabel('')
st.pyplot(plt)

# Footer
st.markdown("### Data visualizations are complete.")
st.markdown("© 2024 Victim Data Analysis. All rights reserved.")
