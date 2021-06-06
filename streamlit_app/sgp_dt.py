import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import logging

import geopandas
from geopy.geocoders import Nominatim

from scipy.spatial.distance import cdist

from streamlit_folium import folium_static
import folium

from shapely.wkt import loads


import shapely

def getStartEnd(start_point,end_point)
    locator = Nominatim(user_agent="http://101.98.38.221:8501")
    if start_point:
        location_s = locator.geocode(start_point)
        x_start = location_s.longitude
        y_start = location_s.latitude
    if end_point:
        location_e = locator.geocode(end_point)
        x_end = location_e.longitude
        y_end = location_e.latitude
        
    df_pts = pd.DataFrame([[x_start,y_start],[x_end,y_end]],columns=['x','y'])
    df_nodes['point'] = [(x, y) for x,y in zip(df_nodes['x'], df_nodes['y'])]
    df_pts['point'] = [(x, y) for x,y in zip(df_pts['x'], df_pts['y'])]
    df_pts['closest'] = [closest_point(x, list(df_nodes['point'])) for x in df_pts['point']]

    start=df_pts.iloc[0]['closest']
    end=df_pts.iloc[1]['closest']
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








def mapIt(start,end,G,SEC=0,CCTV=5.0,LAMP=5.0):
    
    # Creating the weighted network based on the user parameters
    weighted_G, pos, labels = get_weighted_graph(G,SEC,CCTV,LAMP)
    
    # Looking for the closest nodes to the start and end points
    df_nodes = pd.read_csv("../data/SG_nodes.txt", index_col=0)
    df_pts = pd.DataFrame([[x_start,y_start],[x_end,y_end]],columns=['x','y'])
    df_nodes['point'] = [(x, y) for x,y in zip(df_nodes['x'], df_nodes['y'])]
    df_pts['point'] = [(x, y) for x,y in zip(df_pts['x'], df_pts['y'])]
    df_pts['closest'] = [closest_point(x, list(df_nodes['point'])) for x in df_pts['point']]
    start=df_pts.iloc[0]['closest']
    end=df_pts.iloc[1]['closest']

    # Now that we know these nodes, we can find the shorted path
    route=nx.shortest_path(weighted_G,source=start, target=end)
    # And getting the list of the nodes position tuples
    lat,lon = [],[]
    for r in route:
        pt=[r[0],r[1]]
        lat.append(r[1])
        lon.append(r[0]) 
    
    # Now that we know the latlon of the nodes, we can find the polylines linking them
    nodes, lines = [], []
    for r in route:
        nodes.append(dfNodes[dfNodes.Pos == r].iloc[0].ID)
    for k in range(len(nodes)-1):
        R = dfEdges[((dfEdges.TO == nodes[k]) & (dfEdges.FROM == nodes[k+1])) | ((dfEdges.TO == nodes[k+1]) & (dfEdges.FROM == nodes[k]))]
        if len(R):
            lines.append(R)
    df = pd.concat(lines)
    df["geometry"] = df["geometry"].apply(lambda x: loads(x.replace('\'', '')) )
    gdf = geopandas.GeoDataFrame(df, geometry='geometry')

    # Starting the plot
    fig = plot_path(gdf, lat, lon, start, end)

    return fig







def get_weighted_graph(G,security=0,cctv_perf=5.0,lamps_perf=5.0):
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



def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]









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
    # adding the lines joining the nodes
    
    LAT,LON = getLL(gdf)
    
    fig = go.Figure()
     
    
    fig.add_trace(go.Scattermapbox(
        name = "Path",
        mode = "lines",
        lon = long,
        lat = lat,
        marker = {'size': 10},
        line = dict(width = 4.5, color = 'blue')))

    fig.add_trace(go.Scattermapbox(
        name = "Path",
        mode = "lines",
        lon = LON,
        lat = LAT,
        marker = {'size': 10},
        line = dict(width = 4.5, color = 'red')))
        
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
                          'zoom': 13})
    return fig