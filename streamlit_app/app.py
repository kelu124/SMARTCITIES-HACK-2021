import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import logging
st.write(nx.__version__)

# Writing the sidebar
st.sidebar.write("### Parameters")
perf = st.sidebar.slider ( "Green weights" , min_value=0.5 , max_value=2.0 , value=1.0 , step=0.1 , format=None , key=None )
nTirages = st.sidebar.slider (  "Lights" , min_value=5 , max_value=500 , value=10, step=5 , format=None , key=None )
capped = st.sidebar.checkbox('Some value')


@st.cache(allow_output_mutation=True)#allow_output_mutation=True)
def loadShp(path):
    G= nx.readwrite.nx_shp.read_shp(path)
    return G

G = loadShp("data/shp_input_networkx_EPSG32648/split_intersection.shp")

test_G = loadShp("data/StreetsAndLampsAndCCTV.shp")




def get_weighted_graph(G):
    """
    Takes a graph G as input and adds the weights
    """
    weighted_G = nx.Graph()
    for data in G.edges(data=True):
        #logging.info(f'{data}')
        coord1 = data[0]
        coord2 = data[1]
        data_dict = data[2]
        #the new shape file doesn't currently have distences so 
        #if there are no distances we just set the weights to 1
        try:
            distance = data_dict['distance']
            weighted_G.add_edge(coord1,coord2,weight=data_dict['distance'])
        except:
            weighted_G.add_edge(coord1,coord2,weight=1)
    #logging.info(f'{data[0]}')
    #logging.info(f'{data[1]}')
    #logging.info(f'{data[2]}')
    pos = {v:v for v in weighted_G.nodes()}
    labels = nx.get_edge_attributes(weighted_G,'weight')
    return weighted_G, pos, labels


weighted_G, pos, labels = get_weighted_graph(test_G)


#TODO convert physicla address to coordinates
#https://towardsdatascience.com/geocode-with-python-161ec1e62b89

#nx.draw_networkx_nodes(weighted_G,pos, node_size=10,node_color='r')

fig, ax = plt.subplots(frameon=False)
nx.draw_networkx_nodes(weighted_G,pos, node_size=10,node_color='r')
nx.draw_networkx_edges(weighted_G, pos)
st.pyplot(fig)

start=(383474.64370036806,146569.12470285365)
end=(383423.34972914157,145768.7813587437)

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
