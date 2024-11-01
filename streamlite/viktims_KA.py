import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
KA_data = pd.read_csv('/Users/prachit/self/[0] Working/Crime prediction/Crime-data/Data/Victims of KA/VICTIMS_OF_KA_0.csv')
population_data = pd.read_csv('/Users/prachit/self/[0] Working/Crime prediction/Crime-data/population.csv')
population_data = population_data[['Year', 'Population']]
population_data['Population'] = population_data['Population'].astype(int) / 100000

# Data preprocessing
KA_data['STATE/UT'] = KA_data['STATE/UT'].str.upper()
t_states = [state for state in KA_data['STATE/UT'].unique() if state.startswith('TO')]
KA_data = KA_data[~KA_data['STATE/UT'].isin(t_states)]
KA_data = KA_data[~KA_data['STATE/UT'].str.contains('TOTAL')]
year_wise = KA_data.groupby('YEAR')['Grand Total'].sum()
state_wise = KA_data.groupby('STATE/UT')['Grand Total'].sum().sort_values(ascending=False)

# Streamlit setup
st.set_page_config(page_title="Crime Data Analysis", page_icon="🔍", layout="wide")
st.title("Data Analysis")
st.markdown("This app provides an analysis of victim data by year and state.")

# Show raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(KA_data)

# Year-wise total victims plot
st.subheader('Year-wise Total Victims')
year_slider = st.slider('Select Year Range', 
                         min_value=int(year_wise.index.min()), 
                         max_value=int(year_wise.index.max()), 
                         value=(int(year_wise.index.min()), int(year_wise.index.max())))

year_filtered = year_wise[(year_wise.index >= year_slider[0]) & (year_wise.index <= year_slider[1])]
fig_years = go.Figure()
fig_years.add_trace(go.Scatter(x=year_filtered.index, y=year_filtered.values, mode='lines+markers', name='Victims', line=dict(color='orange')))
fig_years.update_layout(title='Year-wise Total Victims', xaxis_title='Year', yaxis_title='Number of Victims', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
st.plotly_chart(fig_years)

# Processing the data
year_wise = KA_data.groupby('YEAR')['Grand Total'].sum()

# Ensure population data index is set correctly
population_data.set_index('Year', inplace=True)

# Year-wise Crime Rate
year_crime_rate = year_wise / population_data['Population']
# Assuming year_crime_rate has been computed correctly
st.subheader('Year-wise Crime Rate')

# Check if the year_crime_rate Series is not empty
if not year_crime_rate.empty:
    fig_crime_rate = go.Figure()
    fig_crime_rate.add_trace(go.Scatter(
        x=year_crime_rate.index,
        y=year_crime_rate.values,
        mode='lines+markers',
        name='Crime Rate',
        line=dict(color='orange')
    ))
    
    fig_crime_rate.update_layout(
        title='Year-wise Crime Rate',
        xaxis_title='Year',
        yaxis_title='Crime Rate (per 100,000)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_crime_rate)
else:
    st.error("No data available for the year-wise crime rate.")

# State-wise total victims plot
st.subheader('State-wise Total Victims')
state_data = pd.DataFrame(state_wise).reset_index()
state_data.columns = ['State/UT', 'Total Victims']
fig_states = px.bar(state_data, 
                    x='Total Victims', 
                    y='State/UT', 
                    orientation='h', 
                    title='Total Victims by State/UT', 
                    color='Total Victims',
                    color_continuous_scale=px.colors.sequential.Reds)
st.plotly_chart(fig_states)

# Victims Distribution by States/UTs
st.subheader('Victims Distribution by States/UTs')
others = state_wise[state_wise < state_wise.sum() * 0.025].sum()
state_wise_filtered = state_wise[state_wise >= state_wise.sum() * 0.025]
state_wise_filtered['Others'] = others

fig_pie = go.Figure(data=[go.Pie(labels=state_wise_filtered.index, values=state_wise_filtered.values, pull=[0.1]*len(state_wise_filtered.index))])
fig_pie.update_layout(title='Victims Distribution by States/UTs', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
st.plotly_chart(fig_pie)

# Trend Analysis over the years
if 'YEAR' in KA_data.columns:
    st.subheader('Trend Analysis of Victims Over Years')
    fig_trend = go.Figure()
    for state in KA_data['STATE/UT'].unique():
        state_data = KA_data[KA_data['STATE/UT'] == state].groupby('YEAR')['Grand Total'].sum().reset_index()
        fig_trend.add_trace(go.Scatter(x=state_data['YEAR'], y=state_data['Grand Total'], mode='lines+markers', name=state))
    fig_trend.update_layout(title='Trend Analysis of Victims Over Years', xaxis_title='Year', yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_trend)

# Trend Analysis over the Purposes
if 'Pupose' in KA_data.columns:
    st.subheader('Trend Analysis of Victims Over Purposes')
    fig_purpose = go.Figure()
    for purpose in KA_data['Pupose'].unique():
        purpose_data = KA_data[KA_data['Pupose'] == purpose].groupby('YEAR')['Grand Total'].sum().reset_index()
        fig_purpose.add_trace(go.Scatter(x=purpose_data['YEAR'], y=purpose_data['Grand Total'], mode='lines+markers', name=purpose))
    fig_purpose.update_layout(title='Trend Analysis of Victims Over Purposes', xaxis_title='Year', yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_purpose)

st.markdown("""
    <style>
    .reportview-container {
        background: #2E2E2E; 
    }
    .sidebar .sidebar-content {
        background: #1A1A1A;  
        color: white;          
    }
    h1, h2, h3, h4, h5, h6, p {
        color: white;          
    }
    </style>
    """, unsafe_allow_html=True)

