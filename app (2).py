import streamlit as st
import plotly.express as px
import pandas as pd
from copy import deepcopy
import json
import time

st.title('Internet over the years')

# cache data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

# import data
df_raw = load_data(path = "data/share-of-individuals-using-the-internet.csv")
df = deepcopy(df_raw)
df = df.rename(columns={"Individuals using the Internet (% of population)" : "Percentage"})

# import geojson
@st.cache_data
def load_geojson(path):
    with open(path, 'r') as f:
        return json.load(f)
geojson = load_geojson("data/countries.geojson")

# select rows with valid geodata and sort it
ent_values = list(df["Code"].unique())
geo_values = [country["properties"]["ISO_A3"] for country in geojson["features"]]
countries = set(ent_values) & set(geo_values)
countries_codes=list(countries) 
valid_codes_df = df[df["Code"].isin(countries_codes)]
sorted_df = valid_codes_df.sort_values(["Year", "Code"]).reset_index(drop=True)

# Placeholder for the message
msg_placeholder = st.empty()
# Show a message
msg_placeholder.success("Did you know that Streamlit reloads every time you play with a widget? Yes, even when you cache. Just give it a second... and stay zen! 😇 ")
# Wait for 3 seconds
time.sleep(10)
# Clear the message
msg_placeholder.empty()

tab1, tab2 = st.tabs(["Countries", "World"])

with tab1:    
    st.write("Here you can see each country's internet usage over time")
    # cache the dictionary with figures for each country
    @st.cache_resource
    def generate_figures():
        country_charts = {}
        for country in sorted_df["Entity"].unique():
            country_df = sorted_df[sorted_df["Entity"] == country]
            fig = px.line(country_df, x= "Year", y="Percentage")
            fig.update_yaxes(range=[0, 100])
            country_charts[country] = fig
        return country_charts
    figs = generate_figures()
    selected_country = st.selectbox('Select a country to see its internet usage over time', sorted(figs.keys()))
    with st.spinner(text='Loading country data...'):
        st.plotly_chart(figs[selected_country], use_container_width=True)

with tab2:
    st.write("Here you can see an animated world map of internet usage over time")
    # cache the animated map
    @st.cache_resource
    def make_animated_map(sorted_df, geojson):
        animated_fig = px.choropleth_map(
            sorted_df[sorted_df["Year"] % 4 == 0],
            geojson=geojson,
            locations="Code",
            featureidkey="properties.ISO_A3",
            color="Percentage",
            animation_frame="Year",
            hover_name="Entity",
            zoom=0.5,
            title="Animated Map by Year",
            labels={'Percentage': 'Internet (%)'},
            width=900, height=650,
            range_color=(0, 100)
        )
        animated_fig.update_layout(
            map_center={"lat": 45, "lon": 0},
            autosize=False,
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        return animated_fig
    if "fig" not in st.session_state:
        with st.spinner("Loading map..."):
            st.session_state["fig"] = make_animated_map(sorted_df, geojson)
    st.plotly_chart(st.session_state["fig"])

# show dataframe
if st.checkbox("Show Dataframe"):
    st.write("This is the dataset:")
    st.dataframe(data=df)