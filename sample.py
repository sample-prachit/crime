import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
from folium import Choropleth
from streamlit_folium import folium_static

# Load India shapefile (update path accordingly)
india_geojson = 'tempdata/india-composite.geojson'

# Sample population data
population_data = {
    'State': ['Maharashtra', 'Uttar Pradesh', 'Bihar', 'West Bengal', 'Madhya Pradesh'],
    'Population': [112374333, 199812341, 104099452, 91276115, 72626809]
}

# Create a DataFrame
df = pd.DataFrame(population_data)

# Load the GeoJSON data
gdf = gpd.read_file(india_geojson)

# Inspect the GeoJSON properties
st.write(gdf.head())  # Display the first few rows to find the right column name
st.write(gdf.columns)  # List all column names

# Replace 'your_geojson_state_column' with the actual name of the state column in your GeoJSON
# For example, if the state names are in a column named 'state_name', use that.
gdf = gdf.merge(df, left_on='state_name', right_on='State', how='left')  # Update as necessary

# Create a Folium map
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Create a choropleth map
Choropleth(
    geo_data=gdf,
    data=gdf['Population'],
    columns=['State', 'Population'],
    key_on='feature.properties.state_name',  # Update based on your GeoJSON structure
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
).add_to(m)

# Streamlit app
st.title("Population Map of India")
st.write("This map shows the population of various states in India with color coding.")
folium_static(m)
