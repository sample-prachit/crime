import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def page1():
    murder_data = pd.read_csv('/Users/prachit/self/[0] Working/Crime prediction/Crime-data/Data/Violent/VICTIM_OF_MURDER_0.csv')
    population_data = pd.read_csv('/Users/prachit/self/[0] Working/Crime prediction/Crime-data/population.csv')
    population_data = population_data[['Year', 'Population']]
    population_data['Population'] = population_data['Population'].astype(int) / 100000

    # Set the title and description of the app
    st.set_page_config(page_title="Crime Data Analysis", page_icon="🔍", layout="wide")
    st.title("Murder Data Analysis")
    # st.markdown("""
    # This app provides an analysis of murder victim data by year, state, gender, and age group.
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

    # Show the raw data
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(murder_data)

    # Processing the data
    year_wise = murder_data.groupby('YEAR')['Total'].sum()
    state_wise = murder_data.groupby('STATE/UT')['Total'].sum().sort_values(ascending=False)

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

    # Processing the data
    year_wise = murder_data.groupby('YEAR')['Total'].sum()

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

    # Individual state selection
    state_select = st.selectbox('Select State/UT for detailed view', state_data['State/UT'])
    if state_select:
        state_victims = state_wise[state_select]
        st.subheader(f'Total Victims in {state_select}: {state_victims}')
        fig_state_detail = go.Figure()
        fig_state_detail.add_trace(go.Bar(x=[state_select], y=[state_victims], marker_color='orange'))
        fig_state_detail.update_layout(title=f'Total Victims in {state_select}', xaxis_title='State/UT', yaxis_title='Number of Victims', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_state_detail)

    # Pie chart for others
    st.subheader('Victims Distribution by States/UTs')
    others = state_wise[state_wise < state_wise.sum() * 0.025].sum()
    state_wise_filtered = state_wise[state_wise >= state_wise.sum() * 0.025]
    state_wise_filtered['Others'] = others

    fig_pie = go.Figure(data=[go.Pie(labels=state_wise_filtered.index, values=state_wise_filtered.values, pull=[0.1]*len(state_wise_filtered.index))])
    fig_pie.update_layout(title='Victims Distribution by States/UTs', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_pie)

    # Gender distribution by age group
    gender_totals = murder_data.drop(columns=['YEAR'], errors='ignore').groupby("GENDER").sum(numeric_only=True).T
    st.subheader('Age Group Distribution by Gender')
    fig_gender_age = go.Figure()
    for gender in gender_totals.columns:
        fig_gender_age.add_trace(go.Bar(x=gender_totals.index, y=gender_totals[gender], name=gender))
    fig_gender_age.update_layout(title='Age Group Distribution by Gender', barmode='stack', xaxis_title='Age Group', yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_gender_age)

    # Trend analysis over the years
    if 'YEAR' in murder_data.columns:
        st.subheader('Trend Analysis of Age Groups by Gender Over Years')
        murder_data_grouped = murder_data.groupby(['YEAR', 'GENDER']).sum(numeric_only=True).reset_index()
        year_gender = st.selectbox('Select Gender for Trend Analysis', murder_data_grouped['GENDER'].unique())
        filtered_data = murder_data_grouped[murder_data_grouped['GENDER'] == year_gender]
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=filtered_data['YEAR'], y=filtered_data['Total'], mode='lines+markers', name=year_gender, line=dict(color='orange')))
        fig_trend.update_layout(title=f'Trend Analysis for {year_gender} Over Years', xaxis_title='Year', yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_trend)

    # Population pyramid
    st.subheader('Population Pyramid')
    pivot_data = murder_data.pivot_table(index='GENDER', 
                                          values=['Upto 10 years', '10-15 years', 
                                                  '15-18 years', '18-30 years', 
                                                  '30-50 years', 'Above 50 years'], 
                                          aggfunc='sum')
    pivot_data = pivot_data.T
    fig_pyramid = go.Figure()
    for gender in pivot_data.columns:
        fig_pyramid.add_trace(go.Bar(x=pivot_data.index, y=pivot_data[gender], name=gender))
    fig_pyramid.update_layout(title='Population Pyramid', barmode='group', xaxis_title='Age Group', yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_pyramid)

    if st.button("Back to Main Page"):
        st.session_state.page = "main"

page1()