import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import logging
import numpy as np
import geopandas as gpd
from geopy.geocoders import Nominatim
from shapely.wkt import loads
from scipy.spatial.distance import cdist
import plotly.graph_objects as go

# Let's brag about digital twins ;)
import sgp_dt as dt

#set wide layout
st.set_page_config(layout="wide")

#run to update datasets
#dt.recomputePrecomputedData()

#read precomputed data
data_obj = dt.loadPrecomputedData()


# Writing the sidebar
start_point = st.sidebar.text_input('Choose starting point...',"Masjid Sultan, Singapore") 
end_point = st.sidebar.text_input('Choose destination...',"Rochor Link Bridge, Singapore") 

prefs = {}
st.sidebar.markdown('# Route Priorities')
st.sidebar.markdown('### Safety')
prefs['cctv'] = 1 if st.sidebar.checkbox('Prioritise CCTV',value = False) else 0
prefs['lamps'] = 1 if st.sidebar.checkbox('Prioritise Street Lighting',value = False) else 0
prefs['tunnels'] = 1 if st.sidebar.checkbox('Avoid Tunnels',value = False) else 0
prefs['pedestrian'] = 1 if st.sidebar.checkbox('Prioritise Footways',value = False) else 0

st.sidebar.markdown('### Mobility')
prefs['stairs'] = 1 if st.sidebar.checkbox('Avoid Stairs',value = False) else 0

st.sidebar.markdown('### Pleasure')
prefs['trees'] = 1 if st.sidebar.checkbox('Prioritise Trees',value = False) else 0

st.sidebar.markdown('---')
st.sidebar.markdown('# Plot Data')
#add boxes to overlay the data
st.sidebar.markdown('### Safety')
plotCCTV = st.sidebar.checkbox('Plot CCTV',value = False)
plotLamps = st.sidebar.checkbox('Plot Street Lights',value = False)
st.sidebar.markdown('### Pleasure')
plotTrees = st.sidebar.checkbox('Plot Trees',value = False)
plotPark = st.sidebar.checkbox('Plot Parks',value = False)

#plotting section
try:
    start,end = dt.getStartEnd(start_point, end_point, data_obj['df_nodes'])
    #create the weighted graph from our penalties
    weighted_G  = dt.modernGraphWeightUpdates(data_obj['G'], prefs)
    #plot the route
    fig = dt.mapIt(start,end,weighted_G)
    if plotTrees:
        dt.add_points_to_figure(fig, *data_obj['tree_ll'], name = 'Trees', color = 'green', opacity =0.5, size = 4)
    if plotLamps:
        dt.add_points_to_figure(fig, *data_obj['lamp_ll'], name = 'Lamps', color = 'orange', opacity =0.7, size = 4)
    if plotPark:
        dt.add_points_to_figure(fig, *data_obj['park_ll'], name = 'Park', color = 'purple', opacity =0.5, size = 4)
    if plotCCTV:
        dt.add_points_to_figure(fig, *data_obj['cctv_ll'], name = 'CCTV', color = 'red', opacity =0.5, size = 4)
    st.write(fig)
except ValueError:
    st.markdown('# Invalid address provided!')

