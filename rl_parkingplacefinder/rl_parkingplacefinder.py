# %%
"""
    parkingplacefinder is an OpenSource python package for the reinforcement learning
    of an agent that searches for a space in a parcking lot

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
"""

import numpy as np
import numpy.random as rnd

import matplotlib.pylab as plt

import time
from datetime import datetime
import math

import networkx as nx
import csv

# %%


class Slot():
    """ 
    Parameters
    ----------
    slot_type: string
        values: 'P','D'
    """

    def __init__(self, id_name, slot_type, slot_state):
        self.id_name = id_name
        self.slot_type = slot_type
        self.slot_state = slot_state


class Filling_Function_Parameters():
    def __init__(self):
        self.utility_function = 'uniform'
        self.uniform_distribution_p_value = 0.7


class Lane_Direction_Parameters():
    """
        directed_driving_lanes : boolean (default: False)
        Indicates if driving lanes are directed, if they are direcrted, lane_direction_function is obligatory
    """
    def __init__(self):
        self.directed_lanes = False
        self.lane_direction_function = "alternates"


class ParkingLot():
    """ Base class for analyzing a directed graphs' bowtie structure values.
    A BowTieNetworkValues stores the number of nodes of each bowtie component 
    and the percent of each component from the whole graph.
    Parameters
    ----------
    w : input integer (default: 10)
        Number of parking slots in dimension width
    l : input integer (default: 1)
        Number of parking slots in dimension lenght
    parking_lane_depth : input integer (default: 2) 
        Single depth = 1, double depth 0 2
    single_depth_outer_lanes: boolean (default: True)
        The first and last lane of the parking are only single depth parking slots
    lane_direction_paramenters : 
        Object of   
    filling_function_parameters : 
        object of 
    """

    def __init__(self, lane_direction_paramenters: Lane_Direction_Parameters, filling_function_parameters: Filling_Function_Parameters, w=10, l=1, parking_lane_depth=2, single_depth_outer_lanes=True):
        self.w = w
        self.l = l
        self.parking_lane_dept = parking_lane_depth
        self.single_depth_outer_lanes = single_depth_outer_lanes
        # helps define structure
        self.lane_direction_paramenters = lane_direction_paramenters
        # create grid
        number_of_lanes = self.get_number_of_lanes()
        self.g = nx.grid_2d_graph(self.w, number_of_lanes, periodic=True)
        self.g = nx.convert_node_labels_to_integers(self.g)

        self.filling_function_parameters = filling_function_parameters

        self.fill_parking_slots()
        self.set_lane_directions()

    def get_number_of_lanes(self):
        """
        Basic function to find total number of aisles (parking and driving) based on the given number of
        parking aisles
        """
        if(self.single_depth_outer_lanes):
            return 2*self.l + 1
        else:
            return math.ceil(self.l/2)

    def set_lane_directions(self):
        if (self.lane_direction_paramenters.directed_lanes):
            if (self.lane_direction_paramenters.lane_direction_function == "alternates"):
                self.set_lane_direction_alternates()

    def set_lane_direction_alternates(self):
        G2 = nx.DiGraph(self.g)
        for edge in G2.edges():
            if edge != tuple(sorted(edge)):
                G2.remove_edge(*edge)

        nx.draw_spectral(G2, node_size=600, node_color='w')

    def plot(self):
        plt.show(self.g)

    def fill_parking_slots(self):
        if self.filling_function_parameters.utility_function == 'uniform':
            return self.fill_parking_uniform()

    def fill_parking_uniform(self):
        for i in self.w*self.l:
            rn = rnd.random()
            if (rn < self.filling_function_parameters.uniform_distribution_p_value):
                print('place occupied')

    def export_parking_lot_data(self):
        """
        Saves parking lot into a csv file
        """
        data = []
        for s in self.g.nodes:
            msg = s + ';'
            data.append(msg)

        with open('parking_lot.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(data)


# %%
