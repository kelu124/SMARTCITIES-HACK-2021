import streamlit as st

import networkx as nx
#import matplotlib.pyplot as plt 
#import pandas as pd
#import seaborn as sns
import logging
#import geopandas as gpd

import osmnx as ox
import networkx as nx
#import plotly.graph_objects as go
import numpy as np
import pickle
import os

#import plotly.express as px

#import geopandas as gpd ## geopandas gdal 
import fiona
from streamlit_folium import folium_static
import folium
from shapely.ops import nearest_points


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


def plot_path(gdf, lat, long, origin_point, destination_point):
    
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
    

    #fig = px.scatter_geo(gdf, locations="geometry" )

    # adding the lines joining the nodes
    fig = go.Figure(go.Scattermapbox(
        name = "Path",
        mode = "lines",
        lon = long,
        lat = lat,
        marker = {'size': 10},
        line = dict(width = 4.5, color = 'blue')))
    
        # adding the lines joining the nodes
        
    fig.add_trace(go.Scattermapbox(
        name = "Parcs",
        mode = "markers",
        lon = gdf.lon,
        lat = gdf.lat,
        marker = {'size': 12, 'color':"blue"}))
    
    
    # adding source marker
    fig.add_trace(go.Scattermapbox(
        name = "Source",
        mode = "markers",
        lon = [origin_point[1]],
        lat = [origin_point[0]],
        marker = {'size': 12, 'color':"red"}))
     
    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name = "Destination",
        mode = "markers",
        lon = [destination_point[1]],
        lat = [destination_point[0]],
        marker = {'size': 12, 'color':'green'}))
    
    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="open-street-map",#stamen-terrain",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                      mapbox = {
                          'center': {'lat': lat_center, 
                          'lon': long_center},
                          'zoom': 15})
    fig.show()

def getNetworkAround(G, lat,lon,distm):
    fn = '../data/osmnx/'+str(lat)+"_"+str(lon)+"_"+str(distm)+'.pickle'
    if not os.path.isfile(fn):
        G = ox.graph_from_point((lat, lon), dist=distm, network_type='all_private') #walk
        with open(fn, 'wb') as f:
            pickle.dump(G, f)
    else:
        with open(fn,"rb") as f:
            G = pickle.load(f)
    return G


# Defining the map boundaries 
def showSGP():
    north, east, south, west = 1.327587854836542, 103.88297275579747, 1.2669696458157633, 103.79947552605792
    # Downloading the map as a graph object 
    G = ox.graph_from_bbox(north, south, east, west, network_type = 'walk',clean_periphery=False)  
    # Plotting the map graph 
    ox.plot_graph(G)

def mapPath(G,origin_point,destination_point,WGT = 'length'):
    # define origin and desination locations 
    #origin_point = (1.278353050783832, 103.84105995326672) 
    #destination_point = (1.2853283229119088, 103.85262989657622)
    # get the nearest nodes to the locations 
    origin_node = ox.distance.nearest_nodes(G, origin_point[1],origin_point[0]) 
    destination_node = ox.distance.nearest_nodes(G, destination_point[1],destination_point[0])
    # printing the closest node id to origin and destination points origin_node, destination_node
    
        # Finding the optimal path 
    route = nx.shortest_path(G, origin_node, destination_node, weight = WGT) 
    # getting coordinates of the nodes
    # we will store the longitudes and latitudes in following list 
    long = [] 
    lat = []  
    for i in route:
        point = G.nodes[i]
        long.append(point['x'])
        lat.append(point['y'])
    #plot_path(gdf,lat, long, origin_point, destination_point)

    #create the base map
    #need to get the tiles and attributes from one of these themes:
    #http://leaflet-extras.github.io/leaflet-providers/preview/
    m = folium.Map(location = [1.2822526633223938, 103.84732075349544], zoom_start = 16,
    tiles='https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png',
    attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')

    # add marker for Liberty Bell
    #tooltip = "Liberty Bell"
    #folium.Marker(
    #    [1.2785088771898996, 103.8413733342337], popup="Liberty Bell", tooltip=tooltip
    #).add_to(m)
    # call to render Folium map in Streamlit 
    m = ox.plot_route_folium(G, route, route_map = m)
    #route_map.save('../data/osmnx/test.html')
    folium_static(m)
