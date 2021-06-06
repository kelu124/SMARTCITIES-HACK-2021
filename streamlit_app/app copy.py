from helpers import *
st.write(nx.__version__)

#old file
#input_file = "data/shp_input_networkx_EPSG32648/split_intersection.shp"
#new file
input_file = "data/StreetsAndLampsAndCCTV.shp"

# Writing the sidebar
st.sidebar.write("### Parameters")
perf = st.sidebar.slider ( "Green weights" , min_value=0.5 , max_value=2.0 , value=1.0 , step=0.1 , format=None , key=None )
nTirages = st.sidebar.slider (  "Lights" , min_value=5 , max_value=500 , value=10, step=5 , format=None , key=None )
capped = st.sidebar.checkbox('Some value')

G = loadShp(input_file)

weighted_G, pos, labels = get_weighted_graph(G)

GDF = gpd.read_file(input_file)

G.graph["crs"] = GDF.crs

origin_point = (1.2822526633223938, 103.84732075349544) 
destination_point = (1.2785088771898996, 103.8413733342337)

mapPath(G,origin_point,destination_point,WGT = 'length')
#TODO convert physicla address to coordinates
#https://towardsdatascience.com/geocode-with-python-161ec1e62b89

#nx.draw_networkx_nodes(weighted_G,pos, node_size=10,node_color='r')

# fig, ax = plt.subplots(frameon=False)
# nx.draw_networkx_nodes(weighted_G,pos, node_size=10,node_color='r')
# nx.draw_networkx_edges(weighted_G, pos)
# st.pyplot(fig)

# start=(383474.64370036806,146569.12470285365)
# end=(383423.34972914157,145768.7813587437)

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
