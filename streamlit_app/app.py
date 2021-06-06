import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import logging

import geopandas
from geopy.geocoders import Nominatim

from scipy.spatial.distance import cdist

# Let's brag about digital twins ;)
import sgp_dt as dt


@st.cache(allow_output_mutation=True)#allow_output_mutation=True)
def loadShp(path):
    # Reading the overall network
    G= nx.readwrite.nx_shp.read_shp(path)
    # Support to network identification
    df_nodes = pd.read_csv("../data/SingaporeStreets/SG_nodes.csv", index_col=0)
    dfNodes = pd.read_csv("../data/dfNodes.csv.zip")
    dfEdges = pd.read_csv("../data/dfEdges.csv.zip") 
    # Additional layers  
    gTrees = gpd.read_file('../data/sTrees.zip') 
    gLamps = gpd.read_file('../data/sLamps.zip') 
    gPark = gpd.read_file('../data/sParks.zip') 
    gCCTV = gpd.read_file('../data/sCCTV.zip') 
    return G, df_nodes, dfNodes, dfEdges, gTrees, gLamps, gPark, gCCTV

# This call should be cached
G, df_nodes, dfNodes, dfEdges, gTrees, gLamps, gPark, gCCTV = loadShp("../data/s3/StreetsLampsCCTVTrees.shp")


# Writing the sidebar

st.sidebar.radio('', ['Walking','Running', 'Cycling'])
st.sidebar.radio('Are you...', ['Local',' Tourist (we optimise your route to help you discover places popular with tourists)'])

start_point = st.sidebar.text_input('Choose starting point...',"Marina Bay, Singapore") 
end_point = st.sidebar.text_input('Choose destination...',"Boat Quay, Singapore") 

security = st.sidebar.checkbox('Security',value = True)
if security:
    cctv_perf = st.sidebar.slider ( "CCTV" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    lamps_perf = st.sidebar.slider ( "Lamps" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None ) 
    pop_perf = st.sidebar.slider ( "Population density" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )

start,end = dt.getStartEnd(start_point,end_point)

fig = dt.mapIt(start,end,G,SEC=security,CCTV=cctv_perf,LAMP=lamps_perf)