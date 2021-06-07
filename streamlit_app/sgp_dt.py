import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import logging
import geopandas
import shapely
from geopy.geocoders import Nominatim
from scipy.spatial.distance import cdist
from shapely.wkt import loads
from functools import wraps
import time

import socket 
if socket.gethostname() == "kelu-e7250":
    import pickle5 as pickle
else:
    import pickle

import geopandas as gpd


def log_time(func):
    """
    decorator to time function and log time
    """
    @wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time() - ts
        logging.info(f"Section {func.__name__} completed in: {te} seconds")
        return result    
    return timed

def cache(*args, **kwargs):
    def decorator(func):
        try:
            __IPYTHON__  # type: ignore
            # We are in a Jupyter environment, so don't apply st.cache
            return func
        except NameError:
            return st.cache(func, *args, **kwargs)

    return decorator

@log_time
def getStartEnd(start_point,end_point,df_nodes,dummy=False):
    """
    takes two text addresses start_point and end_point and returns
    the nearest nodes in the network to those coordinates
    """
    logging.info(f'start_point: {type(start_point)}')
    
    if start_point == '' or end_point == '':
        raise ValueError
    if dummy: #using it because i reached Nominatim limit
        # it selects two points around, artificial I agree but well..
        start_point = (103.855030, 1.2759796) 
        end_point = (103.85019, 1.285865) 
        x_end = end_point[0]
        y_end = end_point[1]
        x_start = start_point[0]
        y_start = start_point[1]

    else:
        # ElSe, we'll be asking for more info online
        locator = Nominatim(user_agent="http://101.98.38.221:8501")
        if start_point:
            location_s = locator.geocode(start_point)
            try:
                x_start = location_s.longitude
                y_start = location_s.latitude
            except:
                st.text('start location bad')
                raise ValueError
        
        if end_point:
            location_e = locator.geocode(end_point)
            try:
                x_end = location_e.longitude
                y_end = location_e.latitude
            except:
                st.text('end location bad')
                raise ValueError

    df_pts = pd.DataFrame([[x_start,y_start],[x_end,y_end]],columns=['x','y'])
    df_nodes['point'] = [(x, y) for x,y in zip(df_nodes['x'], df_nodes['y'])]
    df_pts['point'] = [(x, y) for x,y in zip(df_pts['x'], df_pts['y'])]
    df_pts['closest'] = [closest_point(x, list(df_nodes['point'])) for x in df_pts['point']]
    
    start=df_pts.iloc[0]['closest']
    end=df_pts.iloc[1]['closest']
    return start,end

@log_time
def getLL(gdf):
    """
    Takes a geodataframe gdf and returns the lat and long for the features as 
    lists
    """
    # Used for plotting the real lines from a path
    lats, lons = [], []
    for feature in gdf.geometry:
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            lats = np.append(lats, None)
            lons = np.append(lons, None)
    return lats,lons

@log_time
def addshortest(fig, shortest):
    """
    adds the shortest path from shortest to 
    the plotly mapbox plot fig
    """
    LatShort,LonShort = [],[]
    for r in shortest:
        pt=[r[0],r[1]]
        LatShort.append(r[1])
        LonShort.append(r[0]) 

    fig.add_trace(go.Scattermapbox(
        name = "Shortest pathr",
        mode = "lines",
        lon = LonShort,
        lat = LatShort,
        marker = {'size': 10},
        line = dict(width = 2.5, color = 'green')))

    return fig

@log_time
def mapIt(start,end,weighted_G):
    """
    generates a route through the network by finding the shortest route using our custom weights
    inputs:
    start: start node of route in the network
    end: end node of route in the network
    weighted_G: networkx graph of the network which start and end are in
    """    
    try:
        route = nx.shortest_path(weighted_G ,source=start, target=end, weight = 'weight')
        #shortest = nx.shortest_path(G ,source=start, target=end, weight = "Length")
    except:
        st.markdown("## !! Address not found in network")
        raise ValueError
        
    border = []
    # And getting the list of the nodes position tuples
    lat,lon = [],[]
    for r in route:
        pt=[r[0],r[1]]
        lat.append(r[1])
        lon.append(r[0]) 

    # Starting the plot
    fig = plot_path(lat, lon, start, end)
    #fig = addshortest(fig, shortest)
    return fig, route, border

@log_time
def get_weighted_graph(G,security=1,cctv_pref=5.0,lamps_pref=5.0):
    """
    Takes a graph G as input and adds the weights
    deprecated
    """
    weighted_G = nx.Graph()
    for data in G.edges(data=True):
        data=data
        coord1 = data[0]
        coord2 = data[1]
        if security:
            # ZE formula to tweak
            data[2]['weight']=(data[2]['CCTV50mRE']*cctv_pref)+(data[2]['Lamps50m']*lamps_pref)+data[2]['Length']
            weighted_G.add_edge(coord1,coord2,weight=data[2]['weight'])
        else:
            data[2]['weight']=data[2]['Length']
            weighted_G.add_edge(coord1,coord2,weight=data[2]['weight'])

    pos = {v:v for v in weighted_G.nodes()}
    labels = nx.get_edge_attributes(weighted_G,'weight')
    return weighted_G, pos, labels

@log_time
@cache(allow_output_mutation=True)
def manipulate_base_graph(G):
    """
    convert the tunnels data in the base attribute to be numeric
    run in a seperate function for performance
    """
    for data in G.edges(data=True):    
        data[2]['tunnel_flag'] = 1 if data[2]['tunnel'] == 'T' else 0
        data[2]['steps_flag'] = 1 if data[2]['fclass'] == 'steps' else 0
        data[2]['pedestrian_flag'] = 1 if data[2]['fclass'] in ['pedestrian','footway'] else 0

    return G

@log_time
def modernGraphWeightUpdates(G, prefs):
    """
    Takes a graph G as input and adds the weights
    """
    avoidance_penalty = 100

    weighted_G = nx.Graph()

    for data in G.edges(data=True):
        data=data
        coord1 = data[0]
        coord2 = data[1]
        #convert our tunnel flag from text to int        
        # ZE formula to tweak
        data[2]['weight'] = 1
        #Length is a bad
        data[2]['weight'] = data[2]['weight']  * (data[2]['Length'])
        #these things are all good
        data[2]['weight'] = data[2]['weight'] / (1.0 + 1 * data[2]['CCTV20mRE'] * prefs['cctv'] )
        data[2]['weight'] = data[2]['weight'] / (1.0 + 1*data[2]['Lamps20m']  * prefs['lamps'])
        data[2]['weight'] = data[2]['weight'] / (1.0 + 1*data[2]['Trees20m']  * prefs['trees'])
        data[2]['weight'] = data[2]['weight'] * (1 + data[2]['tunnel_flag']  * prefs['tunnels'] * avoidance_penalty)
        data[2]['weight'] = data[2]['weight'] * (1 +  data[2]['steps_flag'] * prefs['stairs'] * avoidance_penalty)
        data[2]['weight'] = data[2]['weight'] / (1.0 + 1* data[2]['pedestrian_flag']  * prefs['pedestrian'])
        # It is adapted for pedestrians ?
        relativeEase = 1.0
        if data[2]['fclass'] == "pedestrian":
            relativeEase = 0.3
        elif data[2]['fclass'] == "footway":
            relativeEase = 0.3
        elif data[2]['fclass'] == "steps":
            relativeEase = 0.6            
        data[2]['weight'] = data[2]['weight'] * relativeEase

        weighted_G.add_edge(coord1,coord2,weight=data[2]['weight'])

    return weighted_G

@log_time
def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]


@log_time
def plot_path(lat, long, origin_point, destination_point):
    
    """
    Given a list of latitudes and longitudes, origin 
    and destination point, plots a path on a map
    
    Parameters
    ----------
    lat, long: list of latitudes and longitudes
    origin_point, destination_point: co-ordinates of origin
    and destination
    Returns
    -------
    Nothing. Only shows the map.
    """
    # adding the lines joining the nodes
    
    fig = go.Figure(layout = go.Layout(height = 600, width = 1000))

    #add our optimal path
    fig.add_trace(go.Scattermapbox(
        name = "Optimal path - nodes",
        mode = "lines",
        lon = long,
        lat = lat,
        marker = {'size': 10},
        line = dict(width = 4.5, color = 'blue')))

    # adding source marker
    fig.add_trace(go.Scattermapbox(
        name = "Source",
        mode = "markers",
        lon = [origin_point[0]],
        lat = [origin_point[1]],
        marker = {'size': 12, 'color':"black"}))
     
    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name = "Destination",
        mode = "markers",
        lon = [destination_point[0]],
        lat = [destination_point[1]],
        marker = {'size': 12, 'color':'blue'}))
    
    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    # defining the layout using mapbox_style
    #map options:
    #"open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner","stamen-watercolor"
    fig.update_layout(mapbox_style="carto-positron",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                      mapbox = {
                          'center': {'lat': lat_center, 
                          'lon': long_center},
                          'zoom': 16})
    return fig

@log_time
@cache(allow_output_mutation=True)
def get_lat_lons(gdf):
    """
    get the lat and longs for points in a geodataframe
    """
    lats = [p.y for p in gdf.geometry]
    lons = [p.x for p in gdf.geometry]
    return (lats, lons)

@log_time
def add_points_to_figure(fig, lats, lons, name, color, opacity, size):
    """
    adds a set of points to a chart
    """
    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name = name,
        mode = "markers",
        lon = lons,
        lat = lats,
        marker = {'size': size, 'color':color, 'opacity':opacity}))
    return fig

@log_time
def add_gdf_to_figure(fig, gdf, name, color, opacity, size):
    """
    adds a set of points to a chart
    """
    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name = name,
        mode = "markers",
        lon = gdf.x,
        lat = gdf.y,
        marker = {'size': size, 'color':color, 'opacity':opacity}))
    return fig

@log_time
def recomputePrecomputedData():
    """
    simple function to read in data and precompute the data we need for 
    the app
    """

    G, df_nodes,gTrees, gLamps, gPark, gCCTV = loadShp("data/s4/SingaporeLampsCCTVTrees.shp")

    #need to precompute and store these
    #get the coordinate data for our plotting data and cache the results
    tree_ll = get_lat_lons(gTrees)
    lamp_ll = get_lat_lons(gLamps)
    park_ll = get_lat_lons(gPark)
    cctv_ll = get_lat_lons(gCCTV)
    #make some cached data manipulations to our base graph
    G = manipulate_base_graph(G)
   
    data_obj = {
        'df_nodes': df_nodes,
        'tree_ll' : tree_ll,
        'lamp_ll' : lamp_ll,
        'park_ll' : park_ll,
        'cctv_ll' : cctv_ll,
        #"gTrees"  : gTrees,
        #"gLamps"  : gLamps,
        #"gPark"   : gPark,
        #"gCCTV"   : gCCTV,
        'G': G
    }

    with open('data/precomp_data.pickle', 'wb') as handle:
        pickle.dump(data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('precomp_data rewritten')

@log_time
@cache(allow_output_mutation=True)
def loadShp(path):
    """
    Read our data in from source files
    """
    # Reading the overall network
    logging.info(f'reading network')
    G= nx.readwrite.nx_shp.read_shp(path)
    # Support to network identification
    logging.info(f'reading df_nodes')
    df_nodes = pd.read_csv("data/SG_nodes.txt", index_col=0)
    # Some cleaning necessary for csv loadup
    #logging.info(f'reading dfNodes')
    #dfNodes = pd.read_csv("data/dfNodes.csv.zip")
    #logging.info(f'updating dfNodes')
    #dfNodes.Pos = dfNodes.Pos.apply(lambda x: eval(x))
    #logging.info(f'reading dfEdges')
    #dfEdges = pd.read_csv("data/dfEdges.csv.zip") 
    #dfEdges.geometry = dfEdges.geometry.apply(lambda x: loads(x.replace('\'', '')))
    # Additional layers  
    logging.info(f'reading gTrees')
    gTrees = gpd.read_file('data/sTrees.zip') 
    logging.info(f'reading sLamps')
    gLamps = gpd.read_file('data/sLamps.zip') 
    logging.info(f'reading sParks')
    gPark = gpd.read_file('data/sParks.zip') 
    logging.info(f'reading sCCTV')
    gCCTV = gpd.read_file('data/sCCTV.zip') 
    return G, df_nodes, gTrees, gLamps, gPark, gCCTV

@log_time
@cache(allow_output_mutation=True)#allow_output_mutation=True)
def loadPrecomputedData():
    """
    This function loads the precomputed data_obj dictionary to reduce
    runtime performance
    """
    with open('data/precomp_data.pickle', 'rb') as handle:
        data_obj = pickle.load(handle)
    return data_obj
