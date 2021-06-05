# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 12:26:26 2021

@author: BLA88076
"""

import networkx as nx
import matplotlib.pyplot as plt
#import osmnx as ox
import pandas as pd
import seaborn as sns


G= nx.read_shp("C:/Users/BLA88076/Downloads/cc/split_intersection.shp")

weighted_G = nx.Graph()
for data in G.edges(data=True):
   weighted_G.add_edge(data[0],data[1],weight=data[2]['distance'])
pos = {v:v for v in weighted_G.nodes()}
labels = nx.get_edge_attributes(weighted_G,'weight')


#TODO convert physicla address to coordinates
#https://towardsdatascience.com/geocode-with-python-161ec1e62b89

start=(383474.64370036806,146569.12470285365)
end=(383423.34972914157,145768.7813587437)

route=nx.shortest_path(weighted_G,source=start, target=end)

df = pd.DataFrame(columns=['x','y'])
for r in route:
    pt=[r[0],r[1]]
    series = pd.Series(pt, index = df.columns)
    df=df.append(series, ignore_index=True)

sns.scatterplot(data=df, x='x',y='y')

#nx.draw_networkx_nodes(weighted_G,pos, node_size=10,node_color='r')
nx.draw_networkx_edges(weighted_G, pos)
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.xlim(382400, 383300) #This changes and is problem specific
plt.ylim(145650, 147250) #This changes and is problem specific

df.plot.line(x='x',y='y')
plt.xlim(382400, 383300) #This changes and is problem specific
plt.ylim(145650, 147250) #This changes and is problem specific



