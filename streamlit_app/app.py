from helpers import *
#st.write(nx.__version__)

#old file
#input_file = "data/shp_input_networkx_EPSG32648/split_intersection.shp"
#new file
#input_file = "data/StreetsAndLampsAndCCTV.shp"

input_file = '../data/SingaporeStreetLampsCCTV/StreetsAndLampsAndCCTV_EPSG32648.shp'

# https://towardsdatascience.com/find-and-plot-your-optimal-path-using-plotly-and-networkx-in-python-17e75387b873


# Writing the sidebar
st.sidebar.write("### Parameters")
#perf = st.sidebar.slider ( "Green weights" , min_value=0.5 , max_value=2.0 , value=1.0 , step=0.1 , format=None , key=None )
#nTirages = st.sidebar.slider (  "Lights" , min_value=5 , max_value=500 , value=10, step=5 , format=None , key=None )
#capped = st.sidebar.checkbox('Some value')
start_point = st.sidebar.text_input('Choose starting point...',"1.2822526633223938, 103.84732075349544") 
end_point = st.sidebar.text_input('Choose destination...',"1.2785088771898996, 103.8413733342337") 
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
    


# Opening path, streelamps & CCTV
# fname = 'data/SingaporeStreets.zip'
# SLC = gpd.read_file(fname)
# SLC

G = loadShp(input_file)

weighted_G, pos, labels = get_weighted_graph(G, params)


G = getNetworkAround(G, 1.2806, 103.8464, 751)

#need to get the coords from the boxes, problem is they 

origin_point = (1.2822526633223938, 103.84732075349544) 
destination_point = (1.2785088771898996, 103.8413733342337)

start = (float(start.split(',')[0]),float(start.split(',')[1]))
end = (float(end.split(',')[0]), float(end.split(',')[1]))

#need to update to use streamlit-folium
mapPath(G,start,end)
