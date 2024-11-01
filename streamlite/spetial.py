import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def page2():
    # Load the data
    spatial_data = pd.read_csv('/Users/prachit/self/[0] Working/Crime prediction/Crime-data/Data/Spatial Analysis/combined_data.csv')

    st.title("Spatial Crime Data Analysis")
    # st.markdown("""
    # This app provides an analysis of crime data by year, state, and various categories.
    # """)
    # Customize the Streamlit app style
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #2E2E2E;  /* Dark background for the app */
        }
        .sidebar .sidebar-content {
            background: #1A1A1A;  /* Darker sidebar */
            color: white;          /* White text in sidebar */
        }
        h1, h2, h3, h4, h5, h6, p {
            color: white;          /* White text for headers and paragraphs */
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Data preprocessing
    spatial_data['Total'] = spatial_data.iloc[:, 5:].sum(axis=1)
    spatial_data['STATE/UT'] = spatial_data['STATE/UT'].str.title().str.upper()
    spatial_data = spatial_data[~spatial_data['STATE/UT'].str.startswith('TO')]
    spatial_data = spatial_data[~spatial_data['STATE/UT'].isin(['TOTAL (STATES)', 'TOTAL (UTs)', 'TOTAL (ALL)', 'TOTAL (ALL-INDIA)'])]

    # Year-wise and state-wise analysis
    year_wise = spatial_data.groupby('YEAR')['Total'].sum()
    state_wise = spatial_data.groupby('STATE/UT')['Total'].sum().sort_values(ascending=False)

    # Show raw data option
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(spatial_data)

    # Year-wise victims plot
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

    # State-wise victims plot
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

    # Pie chart for others
    st.subheader('Victims Distribution by States/UTs')
    others = state_wise[state_wise < state_wise.sum() * 0.025].sum()
    state_wise_filtered = state_wise[state_wise >= state_wise.sum() * 0.025]
    state_wise_filtered['Others'] = others

    fig_pie = go.Figure(data=[go.Pie(labels=state_wise_filtered.index, values=state_wise_filtered.values, pull=[0.1]*len(state_wise_filtered.index))])
    fig_pie.update_layout(title='Victims Distribution by States/UTs', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_pie)
    if st.button("Back to Main Page"):
        st.session_state.page = "main"  # Set the current page back to main

page2()