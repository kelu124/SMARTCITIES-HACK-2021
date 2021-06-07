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


@st.cache(allow_output_mutation=True)#allow_output_mutation=True)
def loadShp(path):
    # Reading the overall network
    G= nx.readwrite.nx_shp.read_shp(path)
    # Support to network identification
    df_nodes = pd.read_csv("data/SG_nodes.txt", index_col=0)
    # Some cleaning necessary for csv loadup
    dfNodes = pd.read_csv("data/dfNodes.csv.zip")
    dfNodes.Pos = dfNodes.Pos.apply(lambda x: eval(x))
    dfEdges = pd.read_csv("data/dfEdges.csv.zip") 
    dfEdges.geometry = dfEdges.geometry.apply(lambda x: loads(x.replace('\'', '')))
    # Additional layers  
    gTrees = gpd.read_file('data/sTrees.zip') 
    gLamps = gpd.read_file('data/sLamps.zip') 
    gPark = gpd.read_file('data/sParks.zip') 
    gCCTV = gpd.read_file('data/sCCTV.zip') 
    return G, df_nodes, dfNodes, dfEdges, gTrees, gLamps, gPark, gCCTV

# This call should be cached
G, df_nodes, dfNodes, dfEdges, gTrees, gLamps, gPark, gCCTV = loadShp("data/s4/SingaporeLampsCCTVTrees.shp")
#get the coordinate data for our plotting data and cache the results
tree_ll = dt.get_lat_lons(gTrees)
lamp_ll = dt.get_lat_lons(gLamps)
park_ll = dt.get_lat_lons(gPark)
cctv_ll = dt.get_lat_lons(gCCTV)

# Writing the sidebar
start_point = st.sidebar.text_input('Choose starting point...',"Masjid Sultan, Singapore") 
end_point = st.sidebar.text_input('Choose destination...',"Rochor Link Bridge, Singapore") 


st.sidebar.markdown('# Route Priorities')
st.sidebar.markdown('### Safety')
cctv_perf = 1 if st.sidebar.checkbox('Prioritise CCTV',value = False) else 0
lamps_perf = 1 if st.sidebar.checkbox('Prioritise Street Lighting',value = False) else 0

st.sidebar.markdown('### Pleasure')
trees_perf = 1 if st.sidebar.checkbox('Prioritise Trees',value = False) else 0


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
    start,end = dt.getStartEnd(start_point, end_point, df_nodes)
    #create the weighted graph from our penalties
    weighted_G  = dt.modernGraphWeightUpdates(G,cctv_perf,lamps_perf,trees_perf)
    #plot the route
    fig = dt.mapIt(start,end,weighted_G)
    if plotTrees:
        dt.add_points_to_figure(fig, *tree_ll, name = 'Trees', color = 'green', opacity =0.7, size = 6)
    if plotLamps:
        dt.add_points_to_figure(fig, *lamp_ll, name = 'Lamps', color = 'orange', opacity =0.9, size = 6)
    if plotPark:
        dt.add_points_to_figure(fig, *park_ll, name = 'Park', color = 'purple', opacity =0.7, size = 6)
    if plotCCTV:
        dt.add_points_to_figure(fig, *cctv_ll, name = 'CCTV', color = 'red', opacity =0.7, size = 6)
    st.write(fig)
except ValueError:
    st.markdown('# Invalid address provided!')

