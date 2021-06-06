import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import logging
st.write(nx.__version__)

import geopandas
from geopy.geocoders import Nominatim
#import utm
from scipy.spatial.distance import cdist


# Writing the sidebar

st.sidebar.radio('', ['Walking','Running', 'Cycling'])

st.sidebar.radio('Are you...', ['Local',' Tourist (we optimise your route to help you discover places popular with tourists)'])


start_point = st.sidebar.text_input('Choose starting point...',"Marina Bay, Singapore") 
end_point = st.sidebar.text_input('Choose destination...',"Boat Quay, Singapore") 

security = st.sidebar.checkbox('Security',value = True)
if security:
    cctv_perf = st.sidebar.slider ( "CCTV" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    lamps_perf = st.sidebar.slider ( "Lamps" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    trafic_perf = st.sidebar.slider ( "Trafic" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    pop_perf = st.sidebar.slider ( "Population density" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )

if st.sidebar.checkbox('Amenities'):
    st.sidebar.slider ( "Public toilets" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Water fountain" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
if st.sidebar.checkbox('Food'):
    st.sidebar.slider ( "Bubble tea " , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Hawker centres" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Cafes" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Bars and pubs" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
if st.sidebar.checkbox('Attractions'):
    st.sidebar.slider ( "Shopping centers " , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Museums" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Parks" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Events" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )


@st.cache(allow_output_mutation=True)#allow_output_mutation=True)
def loadShp(path):
    G= nx.readwrite.nx_shp.read_shp(path)
    return G

def get_weighted_graph(G):
    """
    Takes a graph G as input and adds the weights
    """
    weighted_G = nx.Graph()
    for data in G.edges(data=True):
        #logging.info(f'{data}')
        data=data
        coord1 = data[0]
        coord2 = data[1]
        if security:
            data[2]['weight']=(data[2]['CCTV50mRE']*cctv_perf)+(data[2]['Lamps50m']*lamps_perf)+data[2]['Length']
            weighted_G.add_edge(coord1,coord2,weight=data[2]['weight'])
        else:
            data[2]['weight']=data[2]['Length']
            weighted_G.add_edge(coord1,coord2,weight=data[2]['weight'])
    #logging.info(f'{data[0]}')
    #logging.info(f'{data[1]}')
    #logging.info(f'{data[2]}')
    pos = {v:v for v in weighted_G.nodes()}
    labels = nx.get_edge_attributes(weighted_G,'weight')
    return weighted_G, pos, labels

def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]

G = loadShp("C:/Users/BLA88076/repos/SMARTCITIES-HACK-2021/data/SingaporeStreets/StreetsLampsCCTVTrees.shp")
weighted_G, pos, labels = get_weighted_graph(G)

locator = Nominatim(user_agent="http://101.98.38.221:8501")
if start_point:
    location_s = locator.geocode(start_point)
    x_start = location_s.longitude
    y_start = location_s.latitude
if end_point:
    location_e = locator.geocode(end_point)
    x_end = location_e.longitude
    y_end = location_e.latitude
    
# df_nodes = pd.DataFrame(columns=['y','x'])
# for n in pos:
#     nxy=[n[0],n[1]]
#     series = pd.Series(nxy, index = df_nodes.columns)
#     df_nodes=df_nodes.append(series, ignore_index=True)
# df_nodes.to_csv("C:/Users/BLA88076/repos/SMARTCITIES-HACK-2021/data/SingaporeStreets/SG_nodes.csv")

df_nodes = pd.read_csv("C:/Users/BLA88076/repos/SMARTCITIES-HACK-2021/data/SingaporeStreets/SG_nodes.csv", index_col=0)
df_pts = pd.DataFrame([[x_start,y_start],[x_end,y_end]],columns=['x','y'])
df_nodes['point'] = [(x, y) for x,y in zip(df_nodes['x'], df_nodes['y'])]
df_pts['point'] = [(x, y) for x,y in zip(df_pts['x'], df_pts['y'])]
df_pts['closest'] = [closest_point(x, list(df_nodes['point'])) for x in df_pts['point']]

start=df_pts.iloc[0]['closest']
end=df_pts.iloc[1]['closest']


route=nx.shortest_path(weighted_G,source=start, target=end)
df = pd.DataFrame(columns=['x','y'])
for r in route:
    pt=[r[0],r[1]]
    series = pd.Series(pt, index = df.columns)
    df=df.append(series, ignore_index=True)


df.plot.line(x='x',y='y')


# fig, ax = plt.subplots(frameon=False)
# nx.draw_networkx_nodes(weighted_G,pos, node_size=10,node_color='r')
# nx.draw_networkx_edges(weighted_G, pos)
# st.pyplot(fig)

# ## Finding shortest path
# if 0:
#     route=nx.shortest_path(weighted_G,source=start, target=end)
#     df = pd.DataFrame(columns=['x','y'])
#     for r in route:
#         pt=[r[0],r[1]]
#         series = pd.Series(pt, index = df.columns)
#         df=df.append(series, ignore_index=True)

#     fig, ax = plt.subplots(frameon=False)
#     sns.scatterplot(data=df, x='x',y='y')
#     st.pyplot(fig)

#     #nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
#     plt.xlim(382400, 383300) #This changes and is problem specific
#     plt.ylim(145650, 147250) #This changes and is problem specific
#     st.pyplot(fig)
