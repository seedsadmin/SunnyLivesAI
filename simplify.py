import collections
import pandas as pd
import geopandas as gpd
import geojson
import functools 
import shapely
import sys
import warnings

import os
from pathlib import Path
from pysal.lib import weights
from tqdm import tqdm
import argparse

class UndirectedGraph:
    """Simple undirected graph.
    Note: stores edges; can not store vertices without edges.
    """

    def __init__(self):
        """Creates an empty `UndirectedGraph` instance.
        """

        # Todo: We might need a compressed sparse row graph (i.e. adjacency array)
        # to make this scale. Let's circle back when we run into this limitation.
        self.edges = collections.defaultdict(set)

    def add_edge(self, s, t):
        """Adds an edge to the graph.
        Args:
          s: the source vertex.
          t: the target vertex.
        Note: because this is an undirected graph for every edge `s, t` an edge `t, s` is added.
        """

        self.edges[s].add(t)
        self.edges[t].add(s)

    def targets(self, v):
        """Returns all outgoing targets for a vertex.
        Args:
          v: the vertex to return targets for.
        Returns:
          A list of all outgoing targets for the vertex.
        """

        return self.edges[v]

    def vertices(self):
        """Returns all vertices in the graph.
        Returns:
          A set of all vertices in the graph.
        """

        return self.edges.keys()

    def empty(self):
        """Returns true if the graph is empty, false otherwise.
        Returns:
          True if the graph has no edges or vertices, false otherwise.
        """
        return len(self.edges) == 0

    def dfs(self, v):
        """Applies a depth-first search to the graph.
        Args:
          v: the vertex to start the depth-first search at.
        Yields:
          The visited graph vertices in depth-first search order.
        Note: does not include the start vertex `v` (except if an edge targets it).
        """

        stack = []
        stack.append(v)

        seen = set()

        while stack:
            s = stack.pop()

            if s not in seen:
                seen.add(s)

                for t in self.targets(s):
                    stack.append(t)

                yield s

    def components(self):
        """Computes connected components for the graph.
        Yields:
          The connected component sub-graphs consisting of vertices; in no particular order.
        """

        seen = set()

        for v in self.vertices():
            if v not in seen:
                component = set(self.dfs(v))
                component.add(v)

                seen.update(component)

                yield component


def simplify_footprints(project_id, region_name, projection):
    str_path = 'uploads/' + str(project_id) +'/prodoutput'
    INPUTS = [str_path]
    OUTPUT = str_path
    REGION = region_name
    UTM=projection

    OUTPUT = Path(OUTPUT)
    INPUT = Path(INPUTS[0])
    #Reading footprints from geojson
    footprints_file = f'{INPUT}/{REGION}_footprints_from_model.geojson'
    print(f'Reading footprints from {footprints_file}')
    #footprints = gpd.read_file(footprints_file).to_crs(epsg=UTM)
    footprints = gpd.read_file(footprints_file)
    print(footprints.head())
    w = weights.DistanceBand.from_dataframe(footprints, 150)
    graph = UndirectedGraph()
    # Iterate through each footprint, find polygons if it intersects & add them as an edge in the Graph
   
    for i, polygon in enumerate(tqdm(footprints.geometry)):
        try:
            # Add a small buffer to the specific polygon
            polygon = polygon.buffer(0.0025)
            graph.add_edge(i, i)
            nearest = w[i]
            # Loop over its neighbouring polygons & check if they intersect with the polygon in picture, add it to the Graph
            for t in nearest:
                if polygon.intersects(footprints.geometry[t]):
                    graph.add_edge(i, t)
        except Exception as error:
            print(i, error)
            continue

    components = list(graph.components())
    print(len(components))

    features = []

    # Iterate through each node of the Graph, merge the neighbouring polygons as one & simplify the geometry by a tolerance factor, default 0.2
    for component in tqdm(components):
        # If the # of neighbours is greater than one, do a union of the neighbouring polygons
        if len(component) > 1:
            merged = (shapely.ops.unary_union([footprints.geometry[v].buffer(.0025) for v in component])).buffer(-1 * 0.0025)
        else:
            merged = footprints.geometry[component.pop()].buffer(0)
            
        try:
            if merged.is_valid:
                # Orient exterior ring of the polygon in counter-clockwise direction.
                if isinstance(merged, shapely.geometry.polygon.Polygon):
                    merged = shapely.geometry.polygon.orient(merged, sign=1.0)
                elif isinstance(merged, shapely.geometry.multipolygon.MultiPolygon):
                    merged = [shapely.geometry.polygon.orient(geom, sign=1.0) for geom in merged.geoms]
                    merged = shapely.geometry.MultiPolygon(merged)
                else:
                    print("Warning: merged feature is neither Polygon nor MultiPoylgon, skipping", file=sys.stderr)
                    continue

                # Simplify the polygons using Douglas Pecker Algorithm
                merged = merged.simplify(tolerance=0.25, preserve_topology=True)
                feature = geojson.Feature(geometry=shapely.geometry.mapping(merged))
                features.append(feature)
            else:
                print(component, "Warning: merged feature is not valid, skipping", file=sys.stderr)
        except Exception as error:
            print(error)
            continue

    print(UTM)
    # Create the GeoJSON file, add its UTM as CRS & save the file
    collection = geojson.FeatureCollection(features)
    collection.crs = {
        "type": "name",
        "properties": {"name": f"epsg:{UTM}"}
    }

    output_file = f'{OUTPUT}/{REGION}_simplified_footprints_from_model.geojson'
    print(f'Saving simplified output to {output_file}')
    with open(f'{output_file}', 'w') as file:
        geojson.dump(collection, file)

    # strs = os.environ['PROJ_LIB']
    # print(strs)

