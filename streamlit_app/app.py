from helpers import *
#st.write(nx.__version__)

#old file
#input_file = "data/shp_input_networkx_EPSG32648/split_intersection.shp"
#new file
input_file = "data/StreetsAndLampsAndCCTV.shp"


# Writing the sidebar
st.sidebar.write("### Parameters")
perf = st.sidebar.slider ( "Green weights" , min_value=0.5 , max_value=2.0 , value=1.0 , step=0.1 , format=None , key=None )
nTirages = st.sidebar.slider (  "Lights" , min_value=5 , max_value=500 , value=10, step=5 , format=None , key=None )
capped = st.sidebar.checkbox('Some value')

# Opening path, streelamps & CCTV
fname = 'data/SingaporeStreets.zip'
#SLC = gpd.read_file(fname)

G = loadShp(input_file)

weighted_G, pos, labels = get_weighted_graph(G)

#GDF = gpd.read_file(input_file)

#xmin, ymin, xmax, ymax = SLC.total_bounds
# Opening trees
# Opening park facilities

# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
# gdf = gpd.read_file('../data/park-facilities/park-facilities-kml.kml', driver='KML')
# gdf['lon'] = gdf['geometry'].x
# gdf['lat'] = gdf['geometry'].y

G = getNetworkAround(G, 1.2806, 103.8464, 751)
# ox.plot_graph(G)

for u,v,d in G.edges(data=True):
    #ID = d["osmid"]
    #d['weight'] = 10 +  1.0 / d['length'] # + d['Lamps50m'] + d["CCTV100mRE"]
    d['weight'] = 1

origin_point = (1.2822526633223938, 103.84732075349544) 
destination_point = (1.2785088771898996, 103.8413733342337)

#need to update to use streamlit-folium
mapPath(G,origin_point,destination_point,WGT = 'length')
