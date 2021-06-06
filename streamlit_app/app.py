import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import logging
st.write(nx.__version__)

# Writing the sidebar
st.sidebar.write("### Parameters")

start_point = st.sidebar.text_input('Choose starting point...',"(383474.64370036806,146569.12470285365)") 
end_point = st.sidebar.text_input('Choose destination...',"(383423.34972914157,145768.7813587437)") 
security_check = st.sidebar.checkbox('Security', value=True)
perf_cctv = st.sidebar.slider ( "CCTV" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
perf_lamps = st.sidebar.slider ( "Lamps" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
security_check = st.sidebar.checkbox('tourist path', value=False)

#start=(383474.64370036806,146569.12470285365)
#end=(383423.34972914157,145768.7813587437)

if start_point:
    start = start_point
if end_point:
    end = end_point
if security_check:
    sec=1
    if perf_cctv:
        cctv=perf_cctv
    if perf_lamps:
        lamps=perf_lamps
else:
    sec=0
    cctv=0
    lamps=0
    

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
        data[2]['weight']=(data[2]['CCTV50mRE']*sec*cctv)+(data[2]['Lamps50m']*sec*lamps)+data[2]['length']
        try:
            weighted_G.add_edge(coord1,coord2,weight=data[2]['weight'])
        except:
            weighted_G.add_edge(coord1,coord2,weight=-999)
    #logging.info(f'{data[0]}')
    #logging.info(f'{data[1]}')
    #logging.info(f'{data[2]}')
    pos = {v:v for v in weighted_G.nodes()}
    labels = nx.get_edge_attributes(weighted_G,'weight')
    return weighted_G, pos, labels

G = loadShp("../data/SingaporeStreetLampsCCTV/StreetsAndLampsAndCCTV_EPSG32648.shp")
weighted_G, pos, labels = get_weighted_graph(G)

fig, ax = plt.subplots(frameon=False)
nx.draw_networkx_nodes(weighted_G,pos, node_size=10,node_color='r')
nx.draw_networkx_edges(weighted_G, pos)
st.pyplot(fig)

## Finding shortest path
if 0:
    route=nx.shortest_path(weighted_G,source=start, target=end)
    df = pd.DataFrame(columns=['x','y'])
    for r in route:
        pt=[r[0],r[1]]
        series = pd.Series(pt, index = df.columns)
        df=df.append(series, ignore_index=True)

    fig, ax = plt.subplots(frameon=False)
    sns.scatterplot(data=df, x='x',y='y')
    st.pyplot(fig)

    #nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.xlim(382400, 383300) #This changes and is problem specific
    plt.ylim(145650, 147250) #This changes and is problem specific
    st.pyplot(fig)
