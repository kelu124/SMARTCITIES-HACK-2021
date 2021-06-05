# Streamlit app

import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

# For debug purposes ;)
st.write(nx.__version__)

# Writing the sidebar
st.sidebar.write("### Parameters")
perf = st.sidebar.slider ( "Green weights" , min_value=0.5 , max_value=2.0 , value=1.0 , step=0.1 , format=None , key=None )
nTirages = st.sidebar.slider (  "Lights" , min_value=5 , max_value=500 , value=10, step=5 , format=None , key=None )
capped = st.sidebar.checkbox('Some value')

## Caching loading the shapefile
@st.cache(allow_output_mutation=True)#allow_output_mutation=True)
def loadShp():
    G= nx.readwrite.nx_shp.read_shp("data/shp_input_networkx_EPSG32648/split_intersection.shp")
    return G

# Loading the loaded shapefile
G = loadShp()

## Adding weights
weighted_G = nx.Graph()
for data in G.edges(data=True):
   weighted_G.add_edge(data[0],data[1],weight=data[2]['distance'])
pos = {v:v for v in weighted_G.nodes()}
labels = nx.get_edge_attributes(weighted_G,'weight')


#TODO convert physicla address to coordinates
#https://towardsdatascience.com/geocode-with-python-161ec1e62b89

# Showing edges and nodes
fig, ax = plt.subplots(frameon=False)
nx.draw_networkx_nodes(weighted_G,pos, node_size=10,node_color='r')
nx.draw_networkx_edges(weighted_G, pos)
st.pyplot(fig)

## Finding shortest path
## Not on the demo as it was somehow making the app crash (streamlit coredumped)
if 0:

    # Start and stops!
    start=(383474.64370036806,146569.12470285365)
    end=(383423.34972914157,145768.7813587437)

    #Finding the route
    route=nx.shortest_path(weighted_G,source=start, target=end)

    df = pd.DataFrame(columns=['x','y'])
    for r in route:
        pt=[r[0],r[1]]
        series = pd.Series(pt, index = df.columns)
        df=df.append(series, ignore_index=True)

    # Plotting the path
    fig, ax = plt.subplots(frameon=False)
    sns.scatterplot(data=df, x='x',y='y')
    st.pyplot(fig)

    plt.xlim(382400, 383300) #This changes and is problem specific 
    st.pyplot(fig)
