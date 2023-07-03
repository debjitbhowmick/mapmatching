# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:20:59 2023

@author: dbho0002
"""

#%%
import geopandas as gpd
import osmnx as ox
# import networkx as nx
# from shapely.geometry import shape
# from shapely.geometry import Point, Polygon
# from shapely.ops import nearest_points
# import geojson
import pandas as pd
# import numpy as np
# import math
import time
from pathlib import Path
# from gpx_converter import Converter
# from leuvenmapmatching.util.gpx import gpx_to_path
import pyproj
# import functions
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings('ignore')
# import ast

# import sys, os, os.path
# os.environ['HTTP_PROXY'] = 'serp-proxy.erc.monash.edu:3128'

#%%

gdf_edges = gpd.read_file('final_network_impcolumns.shp')
gdf_edges = gdf_edges.dropna(subset=['u','v'])
# gdf_edges.dtypes
gdf_edges['u'] = gdf_edges['u'].astype('int64')
gdf_edges['v'] = gdf_edges['v'].astype('int64')
gdf_edges['key']=0


gdf_nodes = gpd.read_file('final_network_nodes.shp')
# gdf_nodes.dtypes

gdf_edges = gdf_edges.set_index(['u','v','key'])
gdf_nodes = gdf_nodes.set_index(['osmid'])


#%%

#Check crs of geodataframes

print(gdf_edges.crs)
print(gdf_nodes.crs)

#Change crs of gdfs if not in projected coordinate system

# gdf_edges = gdf_edges.to_crs("EPSG:32755")
# gdf_nodes = gdf_nodes.to_crs("EPSG:32755")

#%%

#Convert geodataframes to a osmnx graph

G = ox.utils_graph.graph_from_gdfs(gdf_nodes, gdf_edges)
G = G.to_undirected()

# len(G.nodes)
# len(G.edges)

gdf_nodes_2, gdf_edges_2 = ox.graph_to_gdfs(G, nodes=True, edges=True)
gdf_edges_2 = gdf_edges_2.reset_index()

#%%

#Create a map (InMemMap) object
    
from leuvenmapmatching.map.inmem import InMemMap
map_con = InMemMap("myosm", use_latlon=False, use_rtree=False, index_edges=True)
for nid, row in gdf_nodes_2[['x', 'y']].iterrows():
    map_con.add_node(nid, (row['x'], row['y']))
for nid, row in gdf_edges_2[['u', 'v']].iterrows():
    map_con.add_edge(row['u'], row['v'])
    
#%%

#Set the parameters, read the leuvenmapmatching docs for details
#Tune these to optimise mapmatching performance

from leuvenmapmatching.matcher.distance import DistanceMatcher
matcher = DistanceMatcher(map_con,
                          max_dist=500, max_dist_init=500,  # meter
                          min_prob_norm=0.001,
                          non_emitting_length_factor=0.75,
                          obs_noise=200, obs_noise_ne=200,  # meter
                          dist_noise=200,  # meter
                          non_emitting_states=True, 
                          only_edges=True,
                          # expand=True)
                          max_lattice_width=50,
                          avoid_goingback = True)

#%%

#Set the input and projected coordinate reference systems

inProj = pyproj.Proj(init='epsg:4326')
outProj = pyproj.Proj(init='epsg:32755')
#%%

filepath = 'P:\\cycled_study\\processing\\All_biketrips_with_demographicinfo\\All_bike_trips_expanded_byindividualtrips\\'
filepath1 = 'C:\\Users\\dbho0002\Desktop\\mapmatched_trips_2\\'
files_processed = []
files = Path(filepath).glob('*')
count_files = 0

#%%

df_overall_mapmatching_performance = pd.DataFrame(columns=['filename','total_points','downsampled_points','lastidx','process_time'])

for file in files:
    list_performance = []
    count_files = count_files +1
    filename = str(file.name)
    list_performance.append(filename)
    
    # if filename in files_processed:
    if filename in list(df_overall_mapmatching_performance.filename):
        print('File already processed, skipping to the next file \n')
        continue
    
    df = pd.read_csv(filepath + filename)
    list_performance.append(len(df))
    df = df.iloc[::15].reset_index(drop=True) #for high frequency GPS data, I recommend using intermittent data points (every 15 points in this case) for better mapmatching results
    list_performance.append(len(df))
    lon, lat = pyproj.transform(inProj, outProj, list(df['lon']), list(df['lat']))
    track = list(zip(lon, lat))
    df['proj_coordinates'] = track
    
    start = time.process_time()
    states, lastidx = matcher.match(track)
    last_id = len(states)
    print('last_id: '+str(lastidx))
    nodes = matcher.path_pred_onlynodes
    processing_time = time.process_time() - start
    print('Processing time for map matching (seconds): '+str(processing_time))
    list_performance.append(lastidx)
    list_performance.append(processing_time)
    # list_performance.append(0)
    # df_overall_mapmatching_performance.loc[len(df_overall_mapmatching_performance)] = list_performance
    # print(matcher.print_lattice_stats())
    
    df_mapmatching_results = pd.DataFrame()
    
    if lastidx > 0:
        print('Map matching successful')
        print('For file: '+str(filename))
        df['node_pair'] = pd.Series(states)
        df_mapmatching_results['node_pair'] = pd.Series(states)
        list_matched_coordinates = []
        list_dist_obs = []
        list_way_osmid = []
        list_highway = []
        list_cycleway = []
        list_length = []
        for i in range(0,len(df_mapmatching_results)):
            # if math.isnan(df['node_pair'][i]) == False:
            if type(df_mapmatching_results['node_pair'][i]) == tuple:
                match = matcher.lattice_best[i]
                # matched_coordinates = match.edge_m.pi
                dist_obs = match.dist_obs
                G_edges_proj_filtered = gdf_edges_2[(gdf_edges_2['u']==match.edge_m.l1) & (gdf_edges_2['v']==match.edge_m.l2)]
                # matched_point = nearest_points(G_edges_proj_filtered['geometry'].iloc[0], Point(df['proj_coordinates'][i]))[0]
                # matched_coordinates = (matched_point.x, matched_point.y)
                way_osmid = G_edges_proj_filtered['osmid'].iloc[0]
                # way_highway = G_edges_proj_filtered['highway'].iloc[0]
                # way_cycleway = G_edges_proj_filtered['cycleway'].iloc[0]
                # way_length = G_edges_proj_filtered['length'].iloc[0]
    
                # list_matched_coordinates.append(matched_coordinates)
                list_dist_obs.append(dist_obs)
                list_way_osmid.append(way_osmid)
                # list_highway.append(way_highway)
                # list_cycleway.append(way_cycleway)
                # list_length.append(way_length)               
      
        df_mapmatching_results['way_osmid'] = pd.Series(list_way_osmid)
        df_mapmatching_results['dist_obs'] = pd.Series(list_dist_obs)
        # df_mapmatching_results['way_highway'] = pd.Series(list_highway)
        # df_mapmatching_results['way_cycleway'] = pd.Series(list_cycleway)
        # df_mapmatching_results['way_length'] = pd.Series(list_length)
        df_mapmatching_results_2 = df_mapmatching_results.loc[df_mapmatching_results['node_pair'].shift(-1) != df_mapmatching_results['node_pair']]
        # df_mapmatching_results_2.to_csv(filepath1 + filename + '_mapmatched.csv')
        
        print('Number of nodes matched: '+str(len(nodes)))
        print('Number of GPS points: '+str(len(df)))
        print('last_id: '+str(lastidx))
        processing_time = time.process_time() - start
        print('Processing time overall (seconds): '+str(processing_time))
        print('Number of files processed: '+str(count_files) + '\n')
    else:
        print('Map matching unsuccessful')
        print('Moving to the next file \n')
    files_processed.append(filename)
    df_overall_mapmatching_performance.loc[len(df_overall_mapmatching_performance)] = list_performance
    # if count_files == 100:
    #     break

#%%
df_overall_mapmatching_performance['matching_percent'] = 100*(df_overall_mapmatching_performance['lastidx']+1)/df_overall_mapmatching_performance['downsampled_points']
# df_overall_mapmatching_performance.to_csv('Mapmatching_performance.csv')

# df_overall_mapmatching_performance.hist(column='matching_percent',bins=5)
df_lowperf = df_overall_mapmatching_performance[df_overall_mapmatching_performance['matching_percent']<50.0]

#%%