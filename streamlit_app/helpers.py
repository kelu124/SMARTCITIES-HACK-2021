import streamlit as st

import networkx as nx
#import matplotlib.pyplot as plt 
#import pandas as pd
#import seaborn as sns
import logging
import geopandas as gpd

import osmnx as ox
import networkx as nx
#import plotly.graph_objects as go
import numpy as np
import pickle
import os

#import plotly.express as px

import fiona
from streamlit_folium import folium_static
import folium
from shapely.ops import nearest_points
from shapely.geometry import LineString
from shapely.geometry import Point

from scipy.spatial import cKDTree

@st.cache(allow_output_mutation=True)
def loadShp(path):
    G= nx.readwrite.nx_shp.read_shp(path, simplify=False)
    return G

def get_weighted_graph(G, params):
    """
    Takes a graph G as input and adds the weights
    """
    weighted_G = nx.Graph()
    #logging.info(f'{list(G.edges(data=True))[0]}')
    for data in G.edges(data=True):
        coord1 = data[0]
        coord2 = data[1]
        data_dict = data[2]
        #data[2]['weight']=(data[2]['CCTV50mRE']*params['sec']*params['cctv'])+(data[2]['Lamps50m']*params['sec']*params['lamps'])+data[2]['length']
        data[2]['weight']=1
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

def plot_route_folium(
    G,
    route,
    route_map=None,
    popup_attribute=None,
    tiles="cartodbpositron",
    zoom=1,
    fit_bounds=True,
    route_color=None,
    route_width=None,
    route_opacity=None,
    **kwargs,
):
    """
    Plot a route as an interactive Leaflet web map.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    route : list
        the route as a list of nodes
    route_map : folium.folium.Map
        if not None, plot the route on this preexisting folium map object
    popup_attribute : string
        edge attribute to display in a pop-up when an edge is clicked
    tiles : string
        name of a folium tileset
    zoom : int
        initial zoom level for the map
    fit_bounds : bool
        if True, fit the map to the boundaries of the route's edges
    route_color : string
        deprecated, do not use, use kwargs instead
    route_width : numeric
        deprecated, do not use, use kwargs instead
    route_opacity : numeric
        deprecated, do not use, use kwargs instead
    kwargs
        keyword arguments to pass to folium.PolyLine(), see folium docs for
        options (for example `color="#cc0000", weight=5, opacity=0.7`)
    Returns
    -------
    folium.folium.Map
    """
    # deprecation warning
    if route_color is not None:  # pragma: no cover
        kwargs["color"] = route_color
        warn("`route_color` has been deprecated and will be removed: use kwargs instead")
    if route_width is not None:  # pragma: no cover
        kwargs["weight"] = route_width
        warn("`route_width` has been deprecated and will be removed: use kwargs instead")
    if route_opacity is not None:  # pragma: no cover
        kwargs["opacity"] = route_opacity
        warn("`route_opacity` has been deprecated and will be removed: use kwargs instead")

    # create gdf of the route edges in order
    node_pairs = zip(route[:-1], route[1:])
    uvk = ((u, v, min(G[u][v], key=lambda k: G[u][v][k]["length"])) for u, v in node_pairs)
    
    logging.info('testing subgraph')
    logging.info(f'{route}')
    logging.info(f'{G.subgraph(route)}')

    gdf_edges = ox.utils_graph.graph_to_gdfs(G.subgraph(route), nodes=False).loc[uvk]
    return _plot_folium(gdf_edges, route_map, popup_attribute, tiles, zoom, fit_bounds, **kwargs)


def _plot_folium(gdf, m, popup_attribute, tiles, zoom, fit_bounds, **kwargs):
    """
    Plot a GeoDataFrame of LineStrings on a folium map object.
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        a GeoDataFrame of LineString geometries and attributes
    m : folium.folium.Map or folium.FeatureGroup
        if not None, plot on this preexisting folium map object
    popup_attribute : string
        attribute to display in pop-up on-click, if None, no popup
    tiles : string
        name of a folium tileset
    zoom : int
        initial zoom level for the map
    fit_bounds : bool
        if True, fit the map to gdf's boundaries
    kwargs
        keyword arguments to pass to folium.PolyLine()
    Returns
    -------
    m : folium.folium.Map
    """
    # check if we were able to import folium successfully
    if folium is None:  # pragma: no cover
        raise ImportError("folium must be installed to use this optional feature")

    # get centroid
    x, y = gdf.unary_union.centroid.xy
    centroid = (y[0], x[0])

    # create the folium web map if one wasn't passed-in
    if m is None:
        m = folium.Map(location=centroid, zoom_start=zoom, tiles=tiles)

    # identify the geometry and popup columns
    if popup_attribute is None:
        attrs = ["geometry"]
    else:
        attrs = ["geometry", popup_attribute]

    # add each edge to the map
    for vals in gdf[attrs].values:
        params = dict(zip(["geom", "popup_val"], vals))
        pl = _make_folium_polyline(**params, **kwargs)
        pl.add_to(m)

    # if fit_bounds is True, fit the map to the bounds of the route by passing
    # list of lat-lng points as [southwest, northeast]
    if fit_bounds and isinstance(m, folium.Map):
        tb = gdf.total_bounds
        m.fit_bounds([(tb[1], tb[0]), (tb[3], tb[2])])

    return m

def _make_folium_polyline(geom, popup_val=None, **kwargs):
    """
    Turn LineString geometry into a folium PolyLine with attributes.
    Parameters
    ----------
    geom : shapely LineString
        geometry of the line
    popup_val : string
        text to display in pop-up when a line is clicked, if None, no popup
    kwargs
        keyword arguments to pass to folium.PolyLine()
    Returns
    -------
    pl : folium.PolyLine
    """
    # locations is a list of points for the polyline folium takes coords in
    # lat,lng but geopandas provides them in lng,lat so we must reverse them
    locations = [(lat, lng) for lng, lat in geom.coords]

    # create popup if popup_val is not None
    if popup_val is None:
        popup = None
    else:
        # folium doesn't interpret html, so can't do newlines without iframe
        popup = folium.Popup(html=json.dumps(popup_val))

    # create a folium polyline with attributes
    pl = folium.PolyLine(locations=locations, popup=popup, **kwargs)
    return pl


def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    """
    Convert a MultiDiGraph to node and/or edge GeoDataFrames.
    This function is the inverse of `graph_from_gdfs`.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y attributes
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using nodes u and v
    Returns
    -------
    geopandas.GeoDataFrame or tuple
        gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges). gdf_nodes
        is indexed by osmid and gdf_edges is multi-indexed by u, v, key
        following normal MultiDiGraph structure.
    """
    crs = 32648

    if nodes:
        if not G.nodes:  # pragma: no cover
            raise ValueError("graph contains no nodes")

        nodes, data = zip(*G.nodes(data=True))
        logging.info(f'{nodes[0]}')
        #logging.info(f'{data}')
        # convert node x/y attributes to Points for geometry column
        geom = (Point(n) for n in nodes)
        x = [d[0] for d in nodes]
        y = [d[1] for d in nodes]
        data = {'x': x, 'y': y}
        #logging.info(f'{geom}')
        gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs, geometry=list(geom))
        gdf_nodes = gdf_nodes.reset_index().drop(columns=['level_0','level_1'])

    if edges:

        if not G.edges:  # pragma: no cover
            raise ValueError("graph contains no edges")

        u, v, k, data = zip(*G.edges(keys=True, data=True))

        if fill_edge_geometry:

            # subroutine to get geometry for every edge: if edge already has
            # geometry return it, otherwise create it using the incident nodes
            x_lookup = nx.get_node_attributes(G, "x")
            y_lookup = nx.get_node_attributes(G, "y")

            def make_geom(u, v, data, x=x_lookup, y=y_lookup):
                if "geometry" in data:
                    return data["geometry"]
                else:
                    return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))

            geom = map(make_geom, u, v, data)
            gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))

        else:
            gdf_edges = gpd.GeoDataFrame(data)
            if "geometry" not in gdf_edges.columns:
                # if no edges have a geometry attribute, create null column
                gdf_edges["geometry"] = np.nan
            gdf_edges.set_geometry("geometry")
            gdf_edges.crs = crs

        # add u, v, key attributes as index
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        gdf_edges["key"] = k
        gdf_edges.set_index(["u", "v", "key"], inplace=True)

        utils.log("Created edges GeoDataFrame from graph")

    if nodes and edges:
        return gdf_nodes, gdf_edges
    elif nodes:
        return gdf_nodes
    elif edges:
        return gdf_edges
    else:  # pragma: no cover
        raise ValueError("you must request nodes or edges or both")



def nearest_nodes(G, X, Y, return_dist=False):

    logging.info(f'X: {X}')
    logging.info(f'Y: {Y}')

    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        # make coordinates arrays if user passed non-iterable values
        is_scalar = True
        X = np.array([X])
        Y = np.array([Y])

    if np.isnan(X).any() or np.isnan(Y).any():  # pragma: no cover
        raise ValueError("`X` and `Y` cannot contain nulls")
    nodes = graph_to_gdfs(G,nodes=True, edges=False, node_geometry=False)
    logging.info('nodes')
    nodes = nodes[["x", "y"]]
    logging.info(f'{nodes}')

    if ox.projection.is_projected(G.graph["crs"]):
        # if projected, use k-d tree for euclidean nearest-neighbor search
        if cKDTree is None:  # pragma: no cover
            raise ImportError("scipy must be installed to search a projected graph")
        dist, pos = cKDTree(nodes).query(np.array([X, Y]).T, k=1)
        nn = nodes.index[pos]
    else:
        # if unprojected, use ball tree for haversine nearest-neighbor search
        if BallTree is None:  # pragma: no cover
            raise ImportError("scikit-learn must be installed to search an unprojected graph")
        # haversine requires lat, lng coords in radians
        nodes_rad = np.deg2rad(nodes[["y", "x"]])
        points_rad = np.deg2rad(np.array([Y, X]).T)
        dist, pos = BallTree(nodes_rad, metric="haversine").query(points_rad, k=1)
        dist = dist[:, 0] * EARTH_RADIUS_M  # convert radians -> meters
        nn = nodes.index[pos[:, 0]]

    # convert results to correct types for return
    nn = nn.tolist()
    dist = dist.tolist()
    if is_scalar:
        nn = nn[0]
        dist = dist[0]

    if return_dist:
        return nn, dist
    else:
        return nn


def mapPath(G,origin_point,destination_point):


    # get the nearest nodes to the locations 
    # logging.info(f'{origin_point}')
    origin_node = nearest_nodes(G, origin_point[1],origin_point[0]) 
    # logging.info(f'origin node: {origin_node}')
    destination_node = nearest_nodes(G, destination_point[1],destination_point[0])
    # logging.info(f'destination node: {destination_node}')

    #origin_node = list(G.nodes)[0]
    #destination_node = list(G.nodes)[200]

    # Finding the optimal path 
    shortest_route = nx.shortest_path(G, origin_node, destination_node, weight = 'length') 
    best_route = nx.shortest_path(G, origin_node, destination_node, weight = 'weight') 

    #create the base map
    #need to get the tiles and attributes from one of these themes:
    #http://leaflet-extras.github.io/leaflet-providers/preview/
    m = folium.Map(location = [1.2822526633223938, 103.84732075349544], zoom_start = 15,
    tiles='https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png',
    attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')

    tooltip = "Your Location"
    folium.Marker(
       [origin_point[0], origin_point[1]], popup="Your Location", tooltip=tooltip
    ).add_to(m)
    
    logging.info(f'{G}')
    #m = plot_route_folium(G, shortest_route, route_map = m, color='red', tooltip = 'Shortest Path')
    #m = plot_route_folium(G, best_route, route_map = m, color='green', tooltip = 'Safe Path')

    locations = [(lat, lng) for lng, lat in geom.coords]
    pl = folium.PolyLine(locations=locations, color = 'green')
    pl.add_to(m)

    #render the map in streamlit
    folium_static(m)
