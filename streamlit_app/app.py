from helpers import *
#st.write(nx.__version__)

#old file
#input_file = "data/shp_input_networkx_EPSG32648/split_intersection.shp"
#new file
#input_file = "data/StreetsAndLampsAndCCTV.shp"

input_file = '../data/networkx_python_shp/shp_input_networkx_EPSG32648/split_intersection.shp'


# graph = gpd.read_file('../data/SingaporeStreetLampsCCTV/SingaporeStreetLampsCCTV.zip')
# logging.info(f'{graph.crs}')
# graph.set_crs(epsg=4326, inplace=True)
# graph.to_csv('../data/SingaporeStreetLampsCCTV/SingaporeStreetLampsCCTV.csv')

# Writing the sidebar
st.sidebar.write("### Parameters")
#perf = st.sidebar.slider ( "Green weights" , min_value=0.5 , max_value=2.0 , value=1.0 , step=0.1 , format=None , key=None )
#nTirages = st.sidebar.slider (  "Lights" , min_value=5 , max_value=500 , value=10, step=5 , format=None , key=None )
#capped = st.sidebar.checkbox('Some value')
start_point = st.sidebar.text_input('Choose starting point...',"383523.677547, 147619.944594") 
end_point = st.sidebar.text_input('Choose destination...',"383541.655113, 147627.691646") 

security_check = st.sidebar.checkbox('Security', value=True)
perf_cctv = st.sidebar.slider ( "CCTV" , min_value=1 , max_value=1000 , value=10 , step=1 , format=None , key=None )
perf_lamps = st.sidebar.slider ( "Lamps" , min_value=1 , max_value=1000 , value=10 , step=1 , format=None , key=None )
security_check = st.sidebar.checkbox('tourist path', value=False)


if start_point:
    start = start_point

if end_point:
    end = end_point

params = {}

if security_check:
    params['sec']=1
    if perf_cctv:
        params['cctv']=perf_cctv
    if perf_lamps:
        params['lamps']=perf_lamps
else:
    params['sec']=0
    params['cctv']=0
    params['lamps']=0
    
start = (float(start.split(',')[0]),float(start.split(',')[1]))
end = (float(end.split(',')[0]), float(end.split(',')[1]))


#Opening path, streelamps & CCTV
#trees = gpd.read_file('data/trees/trees.shp')
#trees


gpd.read_file(input_file)


G = loadShp(input_file)
logging.info(f'{list(G.edges(data=True))[0]}')
logging.info(f'{list(G.nodes(data=True))[0]}')
G, pos, labels = get_weighted_graph(G, params)

# logging.info(f'{type(G)}')
# logging.info(f'{list(G.nodes)[10]}')
# logging.info(f'{list(G.edges)[10]}')
G = getNetworkAround(G, 1.2806, 103.8464, 751)
# logging.info(f'{type(G)}')
# logging.info(f'{list(G.nodes)[10]}')
# logging.info(f'{list(G.edges)[10]}')



#don't want to use this one
#G = getNetworkAround(G, 1.2806, 103.8464, 751)

#G.graph['crs']=32648

mapPath(G,start,end)