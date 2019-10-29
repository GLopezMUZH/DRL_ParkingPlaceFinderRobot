
# %%
"""
    parkingplacefinder is an OpenSource python package for the reinforcement learning
    of an agent that searches for the most optimal free parking spot in a parking lot

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
"""
#%%
import numpy as np
import numpy.random as rnd

import matplotlib.pylab as plt

import time
from datetime import datetime
import math

import networkx as nx
import csv

from rl_parkingplacefinder.parking_lot import Parking_Lot as Parking_Lot

# %%
"""
class ParkingPlaceFinderModel():
    This is the general class to run a reinforcement learning parking place finder experiment
    Parameters:
    def __init__(self):
        self.id_name = ""
        self.rl_strategy = ""
"""

# %%
class Park_Finder_Agent():
    def __init__(self):
        #row = agent//self.nr_parking_lanes
        #col = agent%self.nr_parking_lanes
        self.id_name = 'hola'


    def find_parking(self, parking_lot :  Parking_Lot):
        # if True: let an agent run through the created parking lot and park in the first available parking spot

        curr_pos = 0 # stores current position of agent, currently always starts on node 0
        status = 'moving' # movement status of agent
        step = 0 # counts number of steps agent takes until it parks
        print("History:")

        while status == 'moving':
            step += 1
            
            # get all neighboring nodes ("possible ways to drive")
            options = list(parking_lot.g.neighbors(curr_pos))
            print(f"Step {step}: Currently on node {curr_pos}")
            
            # check if parking spot is vacant and park on it if yes
            for spot in options:
                if parking_lot.g.nodes[spot]['slot_type'] == 'park' and parking_lot.g.nodes[spot]['occupation'] == 'vacant':
                    curr_pos = spot
                    status = 'parked'
                    print(f"We park in Spot {curr_pos} after {step} steps")
                    break
            
            # update the status of the parking slot
            if (status == 'parked'):
                parking_lot.g.nodes[curr_pos]['occupation'] = 'taken'
                parking_lot.node_color_map[curr_pos]='red'
                # unelegant return but for efficiency
                return curr_pos       
            
            # if no parking spot is vacant: restrict set of options to driveway nodes and randomly continue 
            options = [spot for spot in options if parking_lot.g.nodes[spot]['slot_type'] == 'drive']
            curr_pos = rnd.choice(options) 
                
                





#%%




