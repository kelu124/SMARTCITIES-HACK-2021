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

#add dict of personas with prefilled defaults for the app for the demo
personas = {}


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

# Writing the sidebar
#UseDummyPoints= st.sidebar.checkbox('Dummy values',value = False)
UseDummyPoints = False
if not UseDummyPoints:
    start_point = st.sidebar.text_input('Choose starting point...',"Masjid Sultan, Singapore") 
    end_point = st.sidebar.text_input('Choose destination...',"Rochor Link Bridge, Singapore") 

#security = st.sidebar.checkbox('Security',value = True)
st.sidebar.markdown('# Route Priorities')
security =1
if security:
    #cctv_perf = st.sidebar.slider ( "CCTV" , min_value=0 , max_value=10 , value=0 , step=1 , format=None , key=None )
    cctv_perf = 1 if st.sidebar.checkbox('Prioritise CCTV',value = False) else 0
    #lamps_perf = st.sidebar.slider ( "Lamps" , min_value=0 , max_value=10 , value=0 , step=1 , format=None , key=None )
    lamps_perf = 1 if st.sidebar.checkbox('Prioritise Lamps',value = False) else 0
    #trees_perf = st.sidebar.slider ( "Trees" , min_value=0 , max_value=10 , value=0 , step=1 , format=None , key=None ) 
    trees_perf = 1 if st.sidebar.checkbox('Prioritise Trees',value = False) else 0
    #@todo add type of walk, presence of trees
    
if UseDummyPoints:
    start,end = dt.getStartEnd("nop", "end", df_nodes, dummy= UseDummyPoints)
else:
    start,end = dt.getStartEnd(start_point, end_point, df_nodes, dummy= UseDummyPoints)

st.sidebar.markdown('# Plot Data')
#add boxes to overlay the data
plotTrees = st.sidebar.checkbox('Plot Trees',value = False)
plotLamps = st.sidebar.checkbox('Plot Lamps',value = False)
plotPark = st.sidebar.checkbox('Plot Park',value = False)
plotCCTV = st.sidebar.checkbox('Plot CCTV',value = False)

#create the weighted graph from our penalties
weighted_G  = dt.modernGraphWeightUpdates(G,cctv_perf,lamps_perf,trees_perf)

#plot the route
fig = dt.mapIt(start,end,weighted_G,dfNodes,dfEdges)

#get the coordinate data for our plotting data and cache the results
tree_ll = dt.get_lat_lons(gTrees)
lamp_ll = dt.get_lat_lons(gLamps)
park_ll = dt.get_lat_lons(gPark)
cctv_ll = dt.get_lat_lons(gCCTV)

if plotTrees:
    dt.add_points_to_figure(fig, *tree_ll, name = 'Trees', color = 'green')
if plotLamps:
    dt.add_points_to_figure(fig, *lamp_ll, name = 'Lamps', color = 'orange')
if plotPark:
    dt.add_points_to_figure(fig, *park_ll, name = 'Park', color = 'purple')
if plotCCTV:
    dt.add_points_to_figure(fig, *cctv_ll, name = 'CCTV', color = 'red')

st.write(fig)