#%%
import numpy as np
import numpy.random as rnd

import matplotlib.pylab as plt

import time
from datetime import datetime
import math

import networkx as nx
import csv

#%%
class Slot():
    """
    Parameters
    ----------
    slot_type: string
        values: 'P','D'
    """
    # This class is so far not used for the rest of the code, correct?
    def __init__(self, id_name, slot_type, slot_state):
        self.id_name = id_name
        self.slot_type = slot_type
        self.slot_state = slot_state


class Filling_Function_Parameters():
    def __init__(self, filling_function = 'uniform', uniform_distribution_p_value = 0.7):
        self.filling_function = filling_function
        self.uniform_distribution_p_value = uniform_distribution_p_value


class Lane_Direction_Parameters():
    """
        directed_driving_lanes : boolean (default: False)
        Indicates if driving lanes are directed, if they are directed, lane_direction_function is obligatory
    """
    def __init__(self):
        self.directed_lanes = False
        self.lane_direction_function = "alternates"

#%%
class Parking_Lot():
    """ Base class for analyzing a directed graphs' bowtie structure values.
    A BowTieNetworkValues stores the number of nodes of each bowtie component
    and the percent of each component from the whole graph.
    Parameters
    ----------
    w : input integer (default: 10)
        Number of parking slots in dimension width
    l : input integer (default: 1)
        Number of parking slots in dimension length
    parking_lane_depth : input integer (default: 2)
        Single depth = 1, double depth 0 2
    single_depth_outer_lanes: boolean (default: True)
        Only relevant for double depth parking lanes.
        The first and last lane of the parking are only single depth parking slots
    lane_direction_paramenters :
        Object of
    filling_function_parameters :
        object of
    """

    def __init__(self, lane_direction_paramenters: Lane_Direction_Parameters, filling_function_parameters: Filling_Function_Parameters, nr_parking_slots_per_lane=10, nr_parking_lanes=1, parking_lane_depth=2, single_depth_outer_lanes=True, debug = False, draw_graph = False, show_summary = False):
        self.nr_parking_slots_per_lane = nr_parking_slots_per_lane
        self.nr_parking_lanes = nr_parking_lanes
        self.nr_parking_slots = nr_parking_slots_per_lane*nr_parking_lanes
        self.parking_lane_dept = parking_lane_depth
        self.single_depth_outer_lanes = single_depth_outer_lanes
        self.debug = debug
        # internal variables
        self.node_color_map = []
        self.nr_occupied_parking_slots = 0
        # helps define structure
        self.lane_direction_paramenters = lane_direction_paramenters
        # create grid
        self.slots = [0]*(nr_parking_lanes*nr_parking_slots_per_lane)
        self.nr_slots_per_lane = self.nr_parking_slots_per_lane+2
        self.nr_lanes_total =  self.get_number_of_lanes()
        self.nr_slots_total = self.nr_slots_per_lane*self.nr_lanes_total
        # self.printDebug('number_of_lanes='+str(self.nr_lanes_total))
        # self.printDebug('number_of_slots='+str(self.nr_slots_total))
        self.create_parking_geography()
        # update occupied places
        self.filling_function_parameters = filling_function_parameters
        self.fill_parking_slots()
        # set lane directions TODO
        #self.set_lane_directions()
        #self.export_to_pickle()

        if draw_graph:
            self.plot()

        if show_summary:
            print("nr_parking_slots", self.nr_parking_slots)
            print("nr_occupied_parking_slots = ", self.nr_occupied_parking_slots )
            pct_message = 'percent occupation = ' + f"({(self.nr_occupied_parking_slots/self.nr_parking_slots):.3g})"
            print(pct_message)
            # create occupancy status list including all parking spots
            print('Occupancy Status:')
            for slot_nr_in_graph in range(self.nr_slots_total):
                if self.g.nodes[slot_nr_in_graph]['slot_type'] == 'park':
                    print(f"{slot_nr_in_graph}: {self.g.nodes[slot_nr_in_graph]['occupation']}")


    def create_parking_geography(self):
        # width = number of parking slots + 1 driveway places on each side
        self.g = nx.grid_2d_graph(self.nr_lanes_total, self.nr_slots_per_lane)
#        forbidden = ((1,0),(1,1))
#        self.g.remove_edge(forbidden[0],forbidden[1])
        self.g = nx.convert_node_labels_to_integers(self.g)
        # depth is so far only implemented for parking lanes of depth 1
        if(self.parking_lane_dept==1):
            self.create_parking_single_depth()
        if(self.parking_lane_dept==2):
            self.create_parking_double_depth()


    def create_parking_single_depth(self):
        # only for single depth, we always start with drive path
        slot_nr_in_graph = -1
        for lane in range(self.nr_lanes_total):
            lane_nr = lane+1
            for slot_nr_in_lane in range(self.nr_slots_per_lane):
                slot_nr_in_graph += 1
                # the parking lanes
                if(lane_nr%2==0):
                    # the first and last slot are always drive slots
                    if((slot_nr_in_graph%self.nr_slots_per_lane == 0) or (slot_nr_in_graph%self.nr_slots_per_lane == self.nr_slots_per_lane-1)):
                        self.g.nodes[slot_nr_in_graph]['slot_type'] = 'drive'
                        self.node_color_map.append('grey')
                    else:
                        self.g.nodes[slot_nr_in_graph]['slot_type'] = 'park'
                        self.node_color_map.append('green')
                        self.g.nodes[slot_nr_in_graph]['occupation'] = 'vacant'
                else:
                        self.g.nodes[slot_nr_in_graph]['slot_type'] = 'drive'
                        self.node_color_map.append('grey')

    def create_parking_double_depth(self):
        # only for double depth, we always start with drive path
        slot_nr_in_graph = -1
        for lane in range(self.nr_lanes_total):
            lane_nr = lane+1
            for slot_nr_in_lane in range(self.nr_slots_per_lane):
                slot_nr_in_graph += 1
                # the parking lanes
                # first and last lane are drive lanes
                if(lane_nr==1) or (lane_nr == self.nr_lanes_total) or (slot_nr_in_graph%self.nr_slots_per_lane == self.nr_slots_per_lane-1) or (slot_nr_in_graph%self.nr_slots_per_lane == 0):
                    self.g.nodes[slot_nr_in_graph]['slot_type'] = 'drive'
                    self.node_color_map.append('grey')
                    # every third lane is drive lane
                elif((lane_nr-1)%3==0):
                    self.g.nodes[slot_nr_in_graph]['slot_type'] = 'drive'
                    self.node_color_map.append('grey')
                else:
                    self.g.nodes[slot_nr_in_graph]['slot_type'] = 'park'
                    self.node_color_map.append('green')
                    self.g.nodes[slot_nr_in_graph]['occupation'] = 'vacant'


    def printDebug(self, *val):
        if self.debug:
            """
            with open('file.txt', 'a') as f:
                print(str(datetime.today().strftime("%d-%m-%y %H %M %S")), 'DEBUG', list(val), file=f)
            """
            print(list(val))

    def get_number_of_lanes(self):
        """
        Basic function to find total number of aisles (parking and driving) based on the given number of
        parking aisles
        """
        # single depth parking lanes
        if(self.parking_lane_dept==1):
            return 2*self.nr_parking_lanes + 1
        if(self.parking_lane_dept==2):
            return 2*self.nr_parking_lanes - 1


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
        if(self.g.number_of_nodes() < 1000):
            self.g.nodes.data()
            pos = nx.spring_layout(self.g,iterations=1000)
            nx.draw(self.g, pos=pos,  node_color = self.node_color_map, with_labels=True)
        else:
            print("Graph is too large to be drawn")

    def fill_parking_slots(self):
        if self.filling_function_parameters.filling_function == 'uniform':
            return self.fill_parking_uniform()

    def fill_parking_uniform(self):
        for i in range(self.nr_slots_total):
            if(self.g.nodes[i]['slot_type'] == 'park'):
                rn = rnd.rand()
                if (rn < self.filling_function_parameters.uniform_distribution_p_value):
                    self.node_color_map[i]='red'
                    self.g.nodes[i]['occupation'] = 'taken'
                    # so far only visual indication if parking spot is occupied (red)
                    # otherwise we only have visual indication and know number of occupied spots only on parking lot level
                    self.nr_occupied_parking_slots += 1

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

    def export_to_pickle(self):
        nx.write_gpickle(self.g,'parking_lot_big.gpl')

    def get_env(self):
        return self.g
