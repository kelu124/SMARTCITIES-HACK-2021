#import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import logging
import geopandas
from geopy.geocoders import Nominatim
import numpy as np
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import streamlit as st
#from streamlit_folium import folium_static
#import folium

from shapely.wkt import loads


import shapely

def getStartEnd(start_point,end_point,df_nodes,dummy=False):
   
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

        if end_point:
            location_e = locator.geocode(end_point)

        x_end = location_e.longitude
        y_end = location_e.latitude
        x_start = location_s.longitude
        y_start = location_s.latitude   
                
    df_pts = pd.DataFrame([[x_start,y_start],[x_end,y_end]],columns=['x','y'])
    df_nodes['point'] = [(x, y) for x,y in zip(df_nodes['x'], df_nodes['y'])]
    df_pts['point'] = [(x, y) for x,y in zip(df_pts['x'], df_pts['y'])]
    df_pts['closest'] = [closest_point(x, list(df_nodes['point'])) for x in df_pts['point']]
    
    start=df_pts.iloc[0]['closest']
    end=df_pts.iloc[1]['closest']
    #st.write("#Closest",df_pts)
    #st.write("INIT",start_point,end_point)
    #st.write("APRES",start,end)
    return start,end




def getLL(gdf):
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




def addshortest(fig, shortest):

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



def mapIt(start,end,weighted_G,dfNodes,dfEdges):
    
    # Creating the weighted network based on the user parameters
   
    #st.write(start,end)
    # Now that we know these nodes, we can find the shorted path
    try:
        route    = nx.shortest_path(weighted_G ,source=start, target=end, weight = 'weight')
        #shortest = nx.shortest_path(G ,source=start, target=end, weight = "Length")
    except:
        return "## !! Address not found"
        
    # And getting the list of the nodes position tuples
    lat,lon = [],[]
    for r in route:
        pt=[r[0],r[1]]
        lat.append(r[1])
        lon.append(r[0]) 
    #st.write(route)
    # Now that we know the latlon of the nodes, we can find the polylines linking them
    #nodes, lines = [], []
    #st.write(dfNodes.head(3))
    #for r in route:
    #    NODE = dfNodes[dfNodes.Pos == r]
    ##    if len(NODE):
    #        nodes.append(NODE.iloc[0].ID)

    #st.write(nodes)
    #for k in range(len(nodes)-1):
    #    R = dfEdges[((dfEdges.TO == nodes[k]) & (dfEdges.FROM == nodes[k+1])) | ((dfEdges.TO == nodes[k+1]) & (dfEdges.FROM == nodes[k]))]
    ##    if len(R):
    #        lines.append(R)
    #df = pd.concat(lines)
    #df["geometry"] = df["geometry"].apply(lambda x: loads(x.replace('\'', '')) )
    #gdf = geopandas.GeoDataFrame(df, geometry='geometry')

    # Starting the plot
    fig = plot_path(lat, lon, start, end)
    #fig = addshortest(fig, shortest)
    return fig







def get_weighted_graph(G,security=1,cctv_perf=5.0,lamps_perf=5.0):
    """
    Takes a graph G as input and adds the weights
    """
    weighted_G = nx.Graph()
    for data in G.edges(data=True):
        data=data
        coord1 = data[0]
        coord2 = data[1]
        if security:
            # ZE formula to tweak
            data[2]['weight']=(data[2]['CCTV50mRE']*cctv_perf)+(data[2]['Lamps50m']*lamps_perf)+data[2]['Length']
            weighted_G.add_edge(coord1,coord2,weight=data[2]['weight'])
        else:
            data[2]['weight']=data[2]['Length']
            weighted_G.add_edge(coord1,coord2,weight=data[2]['weight'])

    pos = {v:v for v in weighted_G.nodes()}
    labels = nx.get_edge_attributes(weighted_G,'weight')
    return weighted_G, pos, labels



def modernGraphWeightUpdates(G,cctv_perf=1.0,lamps_perf=1.0,trees_perf=1.0):
    """
    Takes a graph G as input and adds the weights
    """
    weighted_G = nx.Graph()

    for data in G.edges(data=True):
        data=data
        coord1 = data[0]
        coord2 = data[1]

        
        if data[2]['CCTV20mRE'] > 0:
            cctv_present = 1
        else:
            cctv_present = 0

        cctv_present = data[2]['CCTV20mRE']
        # ZE formula to tweak
        data[2]['weight'] = 1
        #Length is a bad
        data[2]['weight'] = data[2]['weight']  *(data[2]['Length'])
        #these things are all good
        data[2]['weight'] = data[2]['weight'] / (1.0 + 0.5 * cctv_present * cctv_perf )
        data[2]['weight'] = data[2]['weight'] / (1.0 + 0.5*data[2]['Lamps20m']  * lamps_perf)
        data[2]['weight'] = data[2]['weight'] / (1.0 + 0.5*data[2]['Trees20m']  * trees_perf) 

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


def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]



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


@st.cache(allow_output_mutation=True)
def get_lat_lons(gdf):
    lats = [p.y for p in gdf.geometry]
    lons = [p.x for p in gdf.geometry]
    return (lats, lons)


def add_points_to_figure(fig, lats, lons, name, color, opacity, size):
    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name = name,
        mode = "markers",
        lon = lons,
        lat = lats,
        marker = {'size': size, 'color':color, 'opacity':opacity}))
    return fig